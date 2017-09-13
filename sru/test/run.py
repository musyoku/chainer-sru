import sys, os
import cupy as xp
import numpy as np
from chainer import links, cuda
import torch
from torch.autograd import Variable
sys.path.append(os.path.join(".."))
from naive_sru import SRU as NaiveSRU
from sru import SRU

@profile
def main():
	gpu_device = 0
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	data_cpu = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	data_gpu = cuda.to_gpu(data_cpu, gpu_device)

	# CPU
	layer = SRU(feature_dimension, feature_dimension)
	for _ in range(100):
		h_cpu = layer(data_cpu)
		layer.reset_state()

	# GPU (define-by-run)
	layer = NaiveSRU(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	for _ in range(100):
		h = layer(data_gpu)
		layer.reset_state()

	# GPU (CUDA Kernel)
	layer = SRU(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	for _ in range(100):
		h_gpu = layer(data_gpu)
		layer.reset_state()

	# GPU (PyTorch)
	with torch.cuda.device(gpu_device):
		from cuda_functional import SRU as PyTorchSRU
		data_gpu_torch = torch.FloatTensor(seq_length, batchsize, feature_dimension).cuda()
		rnn = PyTorchSRU(128, 128,
			num_layers = 1,
			dropout = 0.0,
			rnn_dropout = 0.0,
			use_tanh = 0,
			bidirectional = False
		)
		rnn.cuda()
		for _ in range(100):
			output, hidden = rnn(Variable(data_gpu_torch))

	# LSTM (Chainer)
	layer = links.LSTM(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	for _ in range(100):
		for t in range(seq_length):
			h = layer(data_gpu[..., t])
		layer.reset_state()

	print(h_cpu)
	print(h_gpu)


# @profile
def profile():
	gpu_device = 0
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	# Chainer
	with xp.cuda.Device(gpu_device):
		from sru import SRU
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
