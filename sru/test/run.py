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
def _profile():
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


def autograd(X, W, b, initial_ct=None, use_tanh=False):
	batchsize, feature_dimension, seq_length = X.shape
	if initial_ct is None:
		initial_ct = chainer.Variable(np.zeros((batchsize, feature_dimension), dtype=X.dtype))
	if isinstance(X, chainer.Variable) is False:
		X = chainer.Variable(X)

	U = functions.connection.convolution_2d.convolution_2d(X[:, :, None, :], W[..., None, None])[:, :, 0]
	Z, F, R = functions.split_axis(U, 3, axis=1)
	H = None
	C = None
	bf = functions.broadcast_to(b[:feature_dimension], (batchsize, feature_dimension))
	br = functions.broadcast_to(b[feature_dimension:], (batchsize, feature_dimension))

	ct = initial_ct

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

	return H, C, C[..., -1]

def check_matmul():
	seq_length = 2
	batchsize = 3
	feature_dimension = 4
	X = np.arange(0, batchsize * feature_dimension * seq_length).astype(np.float32).reshape((batchsize, feature_dimension, seq_length))
	W = np.arange(0, feature_dimension ** 2 * 3).astype(np.float32).reshape((feature_dimension * 3, feature_dimension))
	print(X)
	print(W)
	U = np.matmul(W, X)
	print(U)

def check_matmul_grad():
	seq_length = 2
	batchsize = 3
	feature_dimension = 4
	X = chainer.Variable(np.arange(0, batchsize * feature_dimension * seq_length).astype(np.float32).reshape((batchsize, feature_dimension, seq_length)))
	W = chainer.Variable(np.arange(0, feature_dimension ** 2 * 3).astype(np.float32).reshape((feature_dimension * 3, feature_dimension)))

	U = functions.connection.convolution_2d.convolution_2d(X[:, :, None, :], W[..., None, None])[:, :, 0]
	_U = np.matmul(W.data, X.data)
	loss = functions.sum(U)
	loss.backward()
	print(W.data)
	print(X.grad)
	print(xp.sum(W.data, axis=0))

def check_forward(batchsize, feature_dimension, seq_length, use_tanh):
	x_cpu_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length * 3)).astype(np.float32) * 20
	x_gpu_data = cuda.to_gpu(x_cpu_data, gpu_device)

	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		# get true output
		layer = SRU(feature_dimension, feature_dimension, use_tanh=use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu_data[..., :seq_length], layer.W, layer.B, None, layer.use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu_data[..., seq_length:seq_length*2], layer.W, layer.B, last_cell_true, layer.use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu_data[..., seq_length*2:], layer.W, layer.B, last_cell_true, layer.use_tanh)

		# get cuda output
		layer.to_gpu(gpu_device)
		output, cell, last_cell = layer(x_gpu_data[..., :seq_length], None)
		output, cell, last_cell = layer(x_gpu_data[..., seq_length:seq_length*2], last_cell)
		output, cell, last_cell = layer(x_gpu_data[..., seq_length*2:], last_cell)

	threshold = 1e-5
	assert(xp.mean(abs(output_true.data - cuda.to_cpu(output.data))) <= threshold), xp.mean(abs(output_true.data - cuda.to_cpu(output.data)))
	assert(xp.mean(abs(cell_true.data - cuda.to_cpu(cell.data))) <= threshold), xp.mean(abs(cell_true.data - cuda.to_cpu(cell.data)))
	assert(xp.mean(abs(last_cell_true.data - cuda.to_cpu(last_cell.data))) <= threshold), xp.mean(abs(last_cell_true.data - cuda.to_cpu(last_cell.data)))

def check_backward(batchsize, feature_dimension, seq_length, use_tanh):
	x_cpu_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length * 3)).astype(np.float32) * 5
	x_gpu_data = cuda.to_gpu(x_cpu_data, gpu_device)
	x_cpu = chainer.Variable(x_cpu_data)
	x_gpu = chainer.Variable(x_gpu_data)

	with chainer.using_config("train", True):
		# get true output
		layer = SRU(feature_dimension, feature_dimension, use_tanh=use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu[..., :seq_length], layer.W, layer.B, None, layer.use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu[..., seq_length:seq_length*2], layer.W, layer.B, last_cell_true, layer.use_tanh)
		output_true, cell_true, last_cell_true = autograd(x_cpu[..., seq_length*2:], layer.W, layer.B, last_cell_true, layer.use_tanh)

		layer.cleargrads()
		functions.sum(output_true).backward()

		b_grad_true = layer.B.grad.copy()
		w_grad_true = layer.W.grad.copy()
		x_grad_true = x_cpu.grad.copy()

		# print("last_cell_true")
		# print(last_cell_true)

		layer.to_gpu(gpu_device)
		output, cell, last_cell = layer(x_gpu[..., :seq_length], None)
		output, cell, last_cell = layer(x_gpu[..., seq_length:seq_length*2], last_cell)
		output, cell, last_cell = layer(x_gpu[..., seq_length*2:], last_cell)

		# print(np.mean(abs(output_true.data - cuda.to_cpu(output.data))))
		# print(np.mean(abs(cell_true.data - cuda.to_cpu(cell.data))))
		
		layer.cleargrads()
		functions.sum(output).backward()
	# print("last_cell")
	# print(last_cell)

	# print("layer.W.data")
	# print(layer.W.data)

	# print("b_grad")
	# print(b_grad)
	# print("b_grad")
	# print(layer.B.grad)

	# print("w_grad")
	# print(w_grad)
	# print("w_grad")
	# print(layer.W.grad)

	# print("x_grad")
	# print(x_cpu.grad)
	# print("x_grad")
	# print(x_gpu.grad)

	threshold = 1e-3
	assert(xp.mean(abs(b_grad_true - cuda.to_cpu(layer.B.grad))) <= threshold), xp.mean(abs(b_grad_true - cuda.to_cpu(layer.B.grad)))
	assert(xp.mean(abs(w_grad_true - cuda.to_cpu(layer.W.grad))) <= threshold), xp.mean(abs(w_grad_true - cuda.to_cpu(layer.W.grad)))
	assert(xp.mean(abs(x_grad_true - cuda.to_cpu(x_gpu.grad))) <= threshold), xp.mean(abs(x_grad_true - cuda.to_cpu(x_gpu.grad)))

def run_tests():
	batchsize_list = [32, 64, 128]
	seq_length_list = [1, 2, 5, 10, 20]
	feature_dimension_list = [128, 256, 512]
	use_tanh_list = [False, True]

	for batchsize in batchsize_list:
		for seq_length in seq_length_list:
			for feature_dimension in feature_dimension_list:
				for use_tanh in use_tanh_list:
					check_forward(batchsize, feature_dimension, seq_length, use_tanh)
					check_backward(batchsize, feature_dimension, seq_length, use_tanh)
					print((batchsize, feature_dimension, seq_length, use_tanh), "	OK")
	

if __name__ == "__main__":
	check_backward(3, 4, 5, True)
	run_tests()
