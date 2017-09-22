from __future__ import division
from __future__ import print_function
import sys, os, chainer, time, argparse
import seaborn as sns
import numpy as np
import pandas as pd
from chainer import cuda, links, functions
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.append(os.path.join(".."))
from sru import SRU

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-device", "-g", type=int, default=0)
args = parser.parse_args()

cuda.get_device(args.gpu_device).use()

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

def generate_cmap(colors):
	values = range(len(colors))
	vmax = np.ceil(np.max(values))
	color_list = []
	for v, c in zip(values, colors):
		color_list.append( ( v/ vmax, c) )
	return LinearSegmentedColormap.from_list('custom_cmap', color_list)

def plot(df, title):
	sns.set(font_scale=1.5)
	sns.set_style("whitegrid", {"grid.linestyle": "--"})
	df = pd.DataFrame({
		"LSTM": [100, 500],
		"SRU": [100, 500],
		})

	df.index = ["forward","backward"]
	df = df.T
	plt.clf()
	ax = df.plot.barh(stacked=True, cmap=generate_cmap(["#597DBE", "#A0C7F1"]), width=0.2, figsize=(8, 4))
	ax.set_title("l=32, d=256")
	ax.set(xlabel="[ms]")
	plt.tight_layout()
	plt.savefig("benchmark.png")

	
def main():
	batchsize_list = [16, 32, 64]
	seq_length_list = [32, 64]
	feature_dimension_list = [128, 256, 512]
	result_sru = []
	result_lstm = []
	for batchsize in batchsize_list:
		for seq_length in seq_length_list:
			for feature_dimension in feature_dimension_list:
				result_sru += benchmark_sru(batchsize, seq_length, feature_dimension)
				result_lstm += benchmark_lstm(batchsize, seq_length, feature_dimension)

	for sru, lstm in zip(result_sru, result_lstm):
		print(sru, lstm)

if __name__ == '__main__':
	main()