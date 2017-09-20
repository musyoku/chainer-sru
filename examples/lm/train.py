from __future__ import division
from __future__ import print_function
import argparse, sys, os, time, uuid, math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, serializers
sys.path.append(os.path.join("..", ".."))
from sru import SRU
from optim import get_optimizer, decrease_learning_rate, get_current_learning_rate

def Convolution1D(in_channels, out_channels):
	return L.ConvolutionND(1, in_channels, out_channels, 1, stride=1, pad=0, nobias=False)

class RNN(chainer.Chain):
	def __init__(self, vocab_size, num_hidden_units):
		super(RNN, self).__init__()
		self.vocab_size = vocab_size
		self.num_hidden_units = num_hidden_units

		with self.init_scope():
			self.embed = L.EmbedID(vocab_size, num_hidden_units)
			self.l1 = SRU(num_hidden_units, num_hidden_units)
			self.l2 = SRU(num_hidden_units, num_hidden_units)
			self.l3 = Convolution1D(num_hidden_units, vocab_size)

		for param in self.params():
			param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

		self.lc1 = None
		self.lc2 = None

	def reset_state(self):
		self.lc1 = None
		self.lc2 = None

	def __call__(self, x, flatten=False):
		batchsize, seq_length = x.shape
		x = x.reshape((-1,))
		h0 = self.embed(x)
		h0 = F.reshape(h0, (batchsize, seq_length, -1))
		h0 = F.transpose(h0, (0, 2, 1))
		h1, c1, self.lc1 = self.l1(F.dropout(h0, ratio=0.75), self.lc1)
		h2, c2, self.lc2 = self.l2(F.dropout(h1, ratio=0.2), self.lc2)
		out_data = self.l3(F.dropout(h2, 0.75))
		if flatten:
			out_data = F.reshape(F.swapaxes(out_data, 1, 2), (-1, self.vocab_size))
		return out_data

def flatten(t_batch):
	xp = cuda.get_array_module(t_batch)
	return xp.reshape(t_batch, (-1,))

def clear_console():
	printr("")

def printr(string):
	sys.stdout.write("\r\033[2K")
	sys.stdout.write(string)
	sys.stdout.flush()

def save_model(model, filename):
	tmp_filename = str(uuid.uuid4())
	serializers.save_hdf5(tmp_filename, model)
	if os.path.isfile(filename):
		os.remove(filename)
	os.rename(tmp_filename, filename)

def load_model(model, filename):
	if os.path.isfile(filename):
		print("Loading {} ...".format(filename))
		serializers.load_hdf5(filename, model)
		return True
	return False
		

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64, help="Number of examples in each mini-batch")
	parser.add_argument("--seq-length", "-l", type=int, default=35, help="Number of words in each mini-batch")
	parser.add_argument("--total-epochs", "-e", type=int, default=39, help="Number of sweeps over the dataset to train")
	parser.add_argument("--gpu-device", "-g", type=int, default=0, help="GPU ID (negative value indicates CPU)")
	parser.add_argument("--grad-clip", "-gc", type=float, default=5, help="Gradient norm threshold to clip")
	parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="Gradient norm threshold to clip")
	parser.add_argument("--momentum", "-mo", type=float, default=0.9, help="Gradient norm threshold to clip")
	parser.add_argument("--optimizer", "-opt", type=str, default="msgd")
	parser.add_argument("--hidden-units", "-hu", type=int, default=650, help="Number of SRU units in each layer")
	parser.add_argument("--lr-decreasing-start-epoch", "-lrd", type=int, default=175)
	args = parser.parse_args()

	dataset_train, dataset_dev, dataset_test = chainer.datasets.get_ptb_words()
	dataset_dev = np.asarray(dataset_dev, dtype=np.int32)

	vocab_size = max(dataset_train) + 1
	rnn = RNN(vocab_size, args.hidden_units)
	load_model(rnn, "model.hdf5")
	total_iterations_train = len(dataset_train) // (args.seq_length * args.batchsize)

	optimizer = get_optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(rnn)
	optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))

	using_gpu = False
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		rnn.to_gpu()
		using_gpu = True
	xp = rnn.xp

	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss = 0
		epoch_start_time = time.time()

		# training
		for itr in range(total_iterations_train):

			try:
				# sample minbatch
				batch_offsets = np.random.randint(0, len(dataset_train) - args.seq_length - 1, size=args.batchsize)
				x_batch = np.empty((args.batchsize, args.seq_length), dtype=np.int32)
				t_batch = np.empty((args.batchsize, args.seq_length), dtype=np.int32)
				for batch_index, offset in enumerate(batch_offsets):
					sequence = dataset_train[offset:offset + args.seq_length]
					teacher = dataset_train[offset + 1:offset + args.seq_length + 1]
					x_batch[batch_index] = sequence
					t_batch[batch_index] = teacher

				if using_gpu:
					x_batch = cuda.to_gpu(x_batch)
					t_batch = cuda.to_gpu(t_batch)

				t_batch = flatten(t_batch)

				# update model parameters
				with chainer.using_config("train", True):
					rnn.reset_state()
					y_batch = rnn(x_batch, flatten=True)
					loss = F.softmax_cross_entropy(y_batch, t_batch)

					rnn.cleargrads()
					loss.backward()
					optimizer.update()

					sum_loss += float(loss.data)

				printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

			except Exception as e:
				printr("")
				print(str(e))

		save_model(rnn, "model.hdf5")

		# evaluation
		with chainer.no_backprop_mode() and chainer.using_config("train", False):
			x_sequence = dataset_dev[:-1]
			t_sequence = dataset_dev[1:]
			rnn.reset_state()
			total_iterations_dev = math.ceil(len(x_sequence) / args.seq_length)
			offset = 0
			negative_log_likelihood = 0
			for itr in range(total_iterations_dev):
				seq_length = min(offset + args.seq_length, len(x_sequence)) - offset
				x_batch = x_sequence[None, offset:offset + seq_length]
				t_batch = flatten(t_sequence[None, offset:offset + seq_length])

				if using_gpu:
					x_batch = cuda.to_gpu(x_batch)
					t_batch = cuda.to_gpu(t_batch)

				y_batch = rnn(x_batch, flatten=True)
				negative_log_likelihood += float(xp.sum(F.softmax_cross_entropy(y_batch, t_batch, reduce="no").data))

				printr("Computing perplexity ...{:3.0f}% ({}/{})".format((itr + 1) / total_iterations_dev * 100, itr + 1, total_iterations_dev))
				offset += seq_length

			try:
				perplexity = math.exp(negative_log_likelihood / len(dataset_dev))
			except:
				perplexity = None
			
		clear_console()
		print("Epoch {} done in {} min - loss: {:.6f} - log_likelihood: {} - ppl: {} - lr: {:.3f} - total {} min".format(
			epoch + 1, int((time.time() - epoch_start_time) // 60), sum_loss / total_iterations_train, 
			int(-negative_log_likelihood), int(perplexity), get_current_learning_rate(optimizer),
			int((time.time() - training_start_time) // 60)))

		if epoch >= args.lr_decreasing_start_epoch:
			decrease_learning_rate(optimizer, 0.98, 1e-7)

if __name__ == "__main__":
	main()