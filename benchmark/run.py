from __future__ import division
from __future__ import print_function
import sys, os, chainer, time
import numpy as np
from chainer import cuda, links, functions
sys.path.append(os.path.join(".."))
from sru import SRU

gpu_device = 0
cuda.get_device(gpu_device).use()

def benchmark_sru(batchsize, seq_length, feature_dimension, repeat=100):
	layer = SRU(feature_dimension, feature_dimension)
	x_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32) * 5
	x_data = cuda.to_gpu(x_data)
	layer.to_gpu()

	result = []
	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		# forward
		start_time = time.time()
		for i in range(repeat):
			output, cell, last_cell = layer(x_data, None)
		forward_time_mean = (time.time() - start_time) / repeat

	with chainer.using_config("train", True):
		# backward
		start_time = time.time()
		for i in range(repeat):
			output, cell, last_cell = layer(x_data, None)
			layer.cleargrads()
			functions.sum(output).backward()
		backward_time_mean = (time.time() - start_time) / repeat

		result.append((batchsize, seq_length, feature_dimension, forward_time_mean, backward_time_mean))

	return result

def benchmark_lstm(batchsize, seq_length, feature_dimension, repeat=100):
	layer = links.LSTM(feature_dimension, feature_dimension)
	x_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32) * 5
	x_data = cuda.to_gpu(x_data)
	layer.to_gpu()

	result = []
	with chainer.no_backprop_mode() and chainer.using_config("train", False):
		# forward
		start_time = time.time()
		for i in range(repeat):
			layer.reset_state()
			for t in range(seq_length):
				output = layer(x_data[..., t])
		forward_time_mean = (time.time() - start_time) / repeat

	with chainer.using_config("train", True):
		# backward
		start_time = time.time()
		for i in range(repeat):
			layer.reset_state()
			loss = 0
			for t in range(seq_length):
				output = layer(x_data[..., t])
				loss += output
			layer.cleargrads()
			functions.sum(loss).backward()
		backward_time_mean = (time.time() - start_time) / repeat

		result.append((batchsize, seq_length, feature_dimension, forward_time_mean, backward_time_mean))

	return result

def main():
	batchsize_list = [16, 32, 64]
	seq_length_list = [32, 64, 128]
	feature_dimension_list = [128, 256, 512]
	result_sru = []
	result_lstm = []
	for batchsize in batchsize_list:
		for seq_length in seq_length_list:
			for feature_dimension in feature_dimension_list:
				result_sru += benchmark_sru(batchsize, seq_length, feature_dimension)
				result_lstm += benchmark_lstm(batchsize, seq_length, feature_dimension)

if __name__ == '__main__':
	main()