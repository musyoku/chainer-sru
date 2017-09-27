from __future__ import division
from __future__ import print_function
import sys, os, chainer, time, argparse, torch
import seaborn as sns
import numpy as np
import pandas as pd
from chainer import cuda, links, functions
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.append(os.path.join(".."))
from sru import SRU
from sru.test.cuda_functional import SRUCell

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0)
args = parser.parse_args()

cuda.get_device(args.gpu_device).use()

def benchmark_chainer_sru(batchsize, seq_length, feature_dimension, repeat=50):
	layer = SRU(feature_dimension)
	x_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32)
	x_data = cuda.to_gpu(x_data)
	layer.to_gpu()

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

	return forward_time_mean, backward_time_mean

def benchmark_pytorch_sru(batchsize, seq_length, feature_dimension, repeat=50):
	with torch.cuda.device(args.gpu_device):
		layer = SRUCell(feature_dimension, feature_dimension)
		layer.cuda()
		x_data = torch.autograd.Variable(torch.randn(seq_length, batchsize, feature_dimension).cuda())

		# forward
		start_time = time.time()
		for i in range(repeat):
			output, hidden = layer(x_data, None)
		forward_time_mean = (time.time() - start_time) / repeat

		# backward
		start_time = time.time()
		for i in range(repeat):
			output, hidden = layer(x_data, None)
			torch.sum(output).backward()
		backward_time_mean = (time.time() - start_time) / repeat

	return forward_time_mean, backward_time_mean

def benchmark_lstm(batchsize, seq_length, feature_dimension, repeat=50):
	layer = links.LSTM(feature_dimension, feature_dimension)
	x_data = np.random.normal(0, 1, size=(batchsize, feature_dimension, seq_length)).astype(np.float32) * 5
	x_data = cuda.to_gpu(x_data)
	layer.to_gpu()

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

	return forward_time_mean, backward_time_mean

def generate_cmap(colors):
	values = range(len(colors))
	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def plot(df, title, filename):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"grid.linestyle": "--"})
	df.index = ["forward","backward"]
	df = df.T
	plt.clf()
	ax = df.plot.barh(stacked=True, cmap=generate_cmap(["#597DBE", "#A0C7F1"]), width=0.2, figsize=(8, 4))
	ax.set_title(title)
	ax.set(xlabel="[ms]")
	plt.tight_layout()
	plt.savefig("{}.png".format(filename))
	
def main():
	batchsize_list = [16, 32]
	seq_length_list = [16, 32]
	feature_dimension_list = [128, 256, 512, 1024]

	# dummy
	result_chainer = benchmark_chainer_sru(16, 16, 128)
	result_pytorch = benchmark_pytorch_sru(16, 16, 128)
	result_lstm = benchmark_lstm(16, 16, 128)

	for batchsize in batchsize_list:
		for seq_length in seq_length_list:
			for dimension in feature_dimension_list:
				result_chainer = benchmark_chainer_sru(batchsize, seq_length, dimension)
				result_pytorch = benchmark_pytorch_sru(batchsize, seq_length, dimension)
				result_lstm = benchmark_lstm(batchsize, seq_length, dimension)

				forward_chainer, backward_chainer = result_chainer
				forward_pytorch, backward_pytorch = result_pytorch
				forward_lstm, backward_lstm = result_lstm

				df = pd.DataFrame({
					"PyTorch SRU": [forward_pytorch * 1000, backward_pytorch * 1000],
					"Chainer SRU": [forward_chainer * 1000, backward_chainer * 1000],
					})

				title = "l={}, d={}, batchsize={}".format(seq_length, dimension, batchsize)
				plot(df, title, "_" + title)

				df = pd.DataFrame({
					"PyTorch SRU": [forward_pytorch * 1000, backward_pytorch * 1000],
					"Chainer SRU": [forward_chainer * 1000, backward_chainer * 1000],
					"Chainer LSTM": [forward_lstm * 1000, backward_lstm * 1000],
					})
				df = df.ix[:, ["Chainer LSTM", "Chainer SRU", "PyTorch SRU"]]
				plot(df, title, title)

if __name__ == '__main__':
	main()