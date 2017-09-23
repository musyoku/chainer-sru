from __future__ import division
from __future__ import print_function
import argparse, sys, os, time, uuid, math
import numpy as np
import chainer
from chainer import cuda, serializers, functions
sys.path.append(os.path.join("..", ".."))
import sru.nn as nn
from optim import Optimizer

class RNN():
	def __init__(self, vocab_size, ndim_feature, num_layers=2, use_tanh=True, dropout_embedding_softmax=0.75, dropout_rnn=0.2, variational_dropout=False):
		super(RNN, self).__init__()
		self.vocab_size = vocab_size
		self.ndim_feature = ndim_feature
		self.num_layers = num_layers
		self.dropout_softmax = dropout_embedding_softmax
		self.dropout_rnn = dropout_rnn
		self.variational_dropout = variational_dropout

		self.model = nn.Module()

		for l in range(num_layers):
			self.model.add(nn.SRU(ndim_feature, ndim_feature, use_tanh, dropout_rnn if variational_dropout else 0))

		self.model.embed = nn.EmbedID(vocab_size, ndim_feature)
		self.model.fc = nn.Convolution1D(ndim_feature, vocab_size)

		for param in self.model.params():
			if param.name == "W":
				param.data[...] = np.random.normal(0, 0.01, param.data.shape)

		self.reset_state()

	def reset_state(self):
		self.contexts = [None] * self.num_layers

	def __call__(self, x, flatten=False):
		batchsize, seq_length = x.shape
		x = x.reshape((-1,))	# flatten
		in_data = self.model.embed(x)
		in_data = functions.reshape(in_data, (batchsize, seq_length, -1))
		in_data = functions.transpose(in_data, (0, 2, 1))
		in_data = functions.dropout(in_data, self.dropout_softmax)

		for l, sru in enumerate(self.model.layers):
			if self.variational_dropout is False:
				in_data = functions.dropout(in_data, self.dropout_rnn)
			hidden, cell, context = sru(in_data, self.contexts[l])
			in_data = hidden
			self.contexts[l] = context

		out_data = self.model.fc(functions.dropout(in_data, self.dropout_softmax))
		if flatten:
			out_data = functions.reshape(functions.swapaxes(out_data, 1, 2), (-1, self.vocab_size))

		return out_data

	def save(self, filename):
		tmp_filename = str(uuid.uuid4())
		serializers.save_hdf5(tmp_filename, self.model)
		if os.path.isfile(filename):
			os.remove(filename)
		os.rename(tmp_filename, filename)

	def load(self, filename):
		if os.path.isfile(filename):
			print("Loading {} ...".format(filename))
			serializers.load_hdf5(filename, self.model)
			return True
		return False
			
def flatten(t_batch):
	xp = cuda.get_array_module(t_batch)
	return xp.reshape(t_batch, (-1,))

def clear_console():
	printr("")

def printr(string):
	sys.stdout.write("\r\033[2K")
	sys.stdout.write(string)
	sys.stdout.flush()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batchsize", "-b", type=int, default=64)
	parser.add_argument("--seq-length", "-l", type=int, default=35)
	parser.add_argument("--total-epochs", "-e", type=int, default=300)
	parser.add_argument("--gpu-device", "-g", type=int, default=0)
	parser.add_argument("--grad-clip", "-gc", type=float, default=5)
	parser.add_argument("--learning-rate", "-lr", type=float, default=1)
	parser.add_argument("--weight-decay", "-wd", type=float, default=0)
	parser.add_argument("--dropout-embedding-softmax", "-dos", type=float, default=0.75)
	parser.add_argument("--dropout-rnn", "-dor", type=float, default=0.2)
	parser.add_argument("--variational-dropout", "-vdo", dest="variational_dropout", action="store_true", default=False)
	parser.add_argument("--use-tanh", "-tanh", dest="use_tanh", action="store_true", default=False)
	parser.add_argument("--momentum", "-mo", type=float, default=0.9)
	parser.add_argument("--optimizer", "-opt", type=str, default="msgd")
	parser.add_argument("--ndim-feature", "-nf", type=int, default=650)
	parser.add_argument("--num-layers", "-nl", type=int, default=2)
	parser.add_argument("--lr-decay-epoch", "-lrd", type=int, default=20)
	parser.add_argument("--model-filename", "-m", type=str, default="model.hdf5")
	args = parser.parse_args()

	print("#layers={}".format(args.num_layers))
	print("d={}".format(args.ndim_feature))
	print("dropout={}".format("Variational" if args.variational_dropout else "Standard"))

	assert args.num_layers > 0
	assert args.ndim_feature > 0

	dataset_train, dataset_dev, dataset_test = chainer.datasets.get_ptb_words()
	dataset_dev = np.asarray(dataset_dev, dtype=np.int32)

	vocab_size = max(dataset_train) + 1
	rnn = RNN(vocab_size,
		ndim_feature=args.ndim_feature, 
		num_layers=args.num_layers,
		use_tanh=args.use_tanh,
		dropout_embedding_softmax=args.dropout_embedding_softmax, 
		dropout_rnn=args.dropout_rnn, 
		variational_dropout=args.variational_dropout)
	rnn.load(args.model_filename)

	total_iterations_train = len(dataset_train) // (args.seq_length * args.batchsize)

	optimizer = Optimizer(args.optimizer, args.learning_rate, args.momentum)
	optimizer.setup(rnn.model)
	if args.grad_clip > 0:
		optimizer.add_hook(chainer.optimizer.GradientClipping(args.grad_clip))
	if args.weight_decay > 0:
		optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

	using_gpu = False
	if args.gpu_device >= 0:
		cuda.get_device(args.gpu_device).use()
		rnn.model.to_gpu()
		using_gpu = True
	xp = rnn.model.xp

	training_start_time = time.time()
	for epoch in range(args.total_epochs):

		sum_loss = 0
		epoch_start_time = time.time()

		# training
		for itr in range(total_iterations_train):
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
				loss = functions.softmax_cross_entropy(y_batch, t_batch)

				rnn.model.cleargrads()
				loss.backward()
				optimizer.update()

				sum_loss += float(loss.data)
				assert sum_loss == sum_loss, "Encountered NaN!"

			printr("Training ... {:3.0f}% ({}/{})".format((itr + 1) / total_iterations_train * 100, itr + 1, total_iterations_train))

		rnn.save(args.model_filename)

		# evaluation
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

			with chainer.no_backprop_mode() and chainer.using_config("train", False):
				y_batch = rnn(x_batch, flatten=True)
				negative_log_likelihood += float(functions.softmax_cross_entropy(y_batch, t_batch).data) * seq_length

			printr("Computing perplexity ...{:3.0f}% ({}/{})".format((itr + 1) / total_iterations_dev * 100, itr + 1, total_iterations_dev))
			offset += seq_length

		assert negative_log_likelihood == negative_log_likelihood, "Encountered NaN!"
		perplexity = math.exp(negative_log_likelihood / len(dataset_dev))
			
		clear_console()
		print("Epoch {} done in {} sec - loss: {:.6f} - log_likelihood: {} - ppl: {} - lr: {:.3g} - total {} min".format(
			epoch + 1, int(time.time() - epoch_start_time), sum_loss / total_iterations_train, 
			int(-negative_log_likelihood), int(perplexity), optimizer.get_learning_rate(),
			int((time.time() - training_start_time) // 60)))

		if epoch >= args.lr_decay_epoch:
			optimizer.decrease_learning_rate(0.98, final_value=1e-5)

if __name__ == "__main__":
	main()