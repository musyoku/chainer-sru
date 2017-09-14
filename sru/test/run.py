import sys, os, chainer
import cupy as xp
import numpy as np
from chainer import links, cuda, functions
import torch
import torch.autograd
sys.path.append(os.path.join(".."))
from naive_sru import SRU as NaiveSRU
from sru import SRU

gpu_device = 0

# @profile
def profile():
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
			output, hidden = rnn(torch.autograd.Variable(data_gpu_torch))

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


def autograd(X, W, b, ct, use_tanh=False):
	batchsize, feature_dimension, seq_length = X.shape
	if ct is None:
		ct = chainer.Variable(np.zeros((batchsize, feature_dimension), dtype=X.dtype))

	U = functions.connection.convolution_2d.convolution_2d(X[:, :, None, :], W[..., None, None])[:, :, 0]
	R, F, Z = functions.split_axis(U, 3, axis=1)
	H = None
	C = None
	bf = functions.broadcast_to(b[:feature_dimension], (batchsize, feature_dimension))
	br = functions.broadcast_to(b[feature_dimension:], (batchsize, feature_dimension))

	for t in range(seq_length):
		xt = X[..., t]
		zt = Z[..., t]
		ft = functions.sigmoid(F[..., t] + bf)
		rt = functions.sigmoid(R[..., t] + br)

		ct = ft * ct + (1 - ft) * zt
		C = functions.expand_dims(ct, 2) if C is None else functions.concat((C, functions.expand_dims(ct, 2)), axis=2)

		g_ct = ct
		if use_tanh:
			g_ct = functions.tanh(ct)

		ht = rt * g_ct + (1 - rt) * xt
		H = functions.expand_dims(ht, 2) if H is None else functions.concat((H, functions.expand_dims(ht, 2)), axis=2)

	return H, C

def check_backward():
	seq_length = 5
	batchsize = 3
	feature_dimension = 4
	x_cpu_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	x_gpu_data = cuda.to_gpu(x_cpu_data, gpu_device)
	x_cpu = chainer.Variable(x_cpu_data)
	x_gpu = chainer.Variable(x_gpu_data)

	layer = SRU(feature_dimension, feature_dimension)
	layer.reset_state()
	output_true, cell_true = autograd(x_cpu, layer.W, layer.b, layer.ct, layer.use_tanh)
	layer.cleargrads()
	functions.sum(output_true).backward()

	layer.reset_state()
	layer.to_gpu(gpu_device)
	output, cell = layer(x_gpu_data)

	print(np.mean(abs(output_true.data - cuda.to_cpu(output.data))))
	print(np.mean(abs(cell_true.data - cuda.to_cpu(cell.data))))

	print(cell_true)
	# print(layer.b.grad)
	
	layer.cleargrads()
	functions.sum(output).backward()


if __name__ == "__main__":
	check_backward()
