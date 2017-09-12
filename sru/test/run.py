import sys, os
import cupy as xp
from chainer import links
import torch
from torch.autograd import Variable
sys.path.append(os.path.join(".."))
from naive_sru import SRU as NaiveSRU
from sru import SRU

# @profile
def main():
	gpu_device = 0
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	# Chainer
	with xp.cuda.Device(gpu_device):
		data = xp.random.normal(0, 1, size=(batchsize, seq_length, feature_dimension)).astype(xp.float32)
		# SRU
		layer = SRU(128, 128)
		layer.to_gpu(gpu_device)
		for _ in range(10):
			h = layer(data)
			layer.reset_state()
		# SRU
		layer = NaiveSRU(128, 128)
		layer.to_gpu(gpu_device)
		for _ in range(10):
			h = layer(data.transpose((0, 2, 1)))
			layer.reset_state()
		# LSTM
		layer = links.LSTM(128, 128)
		layer.to_gpu(gpu_device)
		for _ in range(10):
			for t in range(seq_length):
				h = layer(data[:, t])
			layer.reset_state()
	# PyTorch
	with torch.cuda.device(gpu_device):
		from cuda_functional import SRU as PyTorchSRU
		rnn = PyTorchSRU(128, 128,
			num_layers = 1,
			dropout = 0.0,
			rnn_dropout = 0.0,
			use_tanh = 0,
			bidirectional = False
		)
		rnn.cuda()
		for _ in range(10):
			output, hidden = rnn(Variable(torch.FloatTensor(seq_length, batchsize, feature_dimension).cuda()))

if __name__ == "__main__":
	main()
