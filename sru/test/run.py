import sys, os
import cupy as xp
import numpy as np
from chainer import links, cuda
import torch
from torch.autograd import Variable
sys.path.append(os.path.join(".."))
from naive_sru import SRU as NaiveSRU
from sru import SRU

# @profile
def profile():
	gpu_device = 1
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	data_cpu = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	data_gpu = cuda.to_gpu(data_cpu, gpu_device)

	# CPU
	layer = SRU(feature_dimension, feature_dimension)
	for _ in range(100):
		h_cpu, c_cpu = layer(data_cpu)
		layer.reset_state()

	# GPU (define-by-run)
	layer = NaiveSRU(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	for _ in range(100):
		h, c = layer(data_gpu)
		layer.reset_state()

	# GPU (CUDA Kernel)
	layer = SRU(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	for _ in range(100):
		h_gpu, c_gpu = layer(data_gpu)
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


def check_outputs():
	gpu_device = 0
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	data_cpu = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	data_gpu = cuda.to_gpu(data_cpu, gpu_device)

	# CPU
	layer = SRU(feature_dimension, feature_dimension)
	h_cpu, c_cpu = layer(data_cpu)
	layer.reset_state()

	# GPU
	layer.to_gpu(gpu_device)
	h_gpu, c_gpu = layer(data_gpu)
	layer.reset_state()

	print(np.mean(abs(c_cpu.data - cuda.to_cpu(c_gpu.data))))
	print(np.mean(abs(h_cpu.data - cuda.to_cpu(h_gpu.data))))

def check_backward():
	gpu_device = 0
	seq_length = 50
	batchsize = 48
	feature_dimension = 128
	data_cpu = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	data_gpu = cuda.to_gpu(data_cpu, gpu_device)

	layer = SRU(feature_dimension, feature_dimension)
	layer.to_gpu(gpu_device)
	output, cell = layer(data_gpu)
	output.backward()

if __name__ == "__main__":
	check_backward()
