import math
import numpy as np
import chainer
from chainer import cuda, Variable, function, link, functions, links, initializers
from chainer.utils import type_check
from chainer.links import EmbedID, Linear, BatchNormalization, ConvolutionND, Bias

def Convolution1D(in_channels, out_channels, ksize, stride=1, pad=0, initialW=None):
	return ConvolutionND(1, in_channels, out_channels, ksize, stride=stride, pad=pad, initialW=initialW)

class SRU(link.Chain):
	def __init__(self, in_channels, out_channels, use_tanh=False):
		if True:
			initialW = initializers.Uniform(scale=0.05)
		else:
			wstd = math.sqrt(in_channels)
			initialW = initializers.Normal(wstd)
		super().__init__(W=Convolution1D(in_channels, 3 * out_channels, 1, stride=1, pad=0, initialW=initialW),
			bf=Bias(shape=(out_channels,)), br=Bias(shape=(out_channels,)))
		self.use_highway_connections = in_channels == out_channels
		self.use_tanh = use_tanh
		self.reset_state()

	def __call__(self, X):
		WX = self.W(X)
		Z, F, R = functions.split_axis(WX, 3, axis=1)

		length = X.shape[2]
		for t in range(length):
			xt = X[..., t]
			zt = Z[..., t]
			ft = self.bf(F[..., t])
			rt = self.br(F[..., t])

			if self.ct is None:
				self.ct = (1 - ft) * zt
			else:
				self.ct = ft * self.ct + (1 - ft) * zt

			if self.use_tanh:
				self.ct = functions.tanh(self.ct)

			self.ht = rt * self.ct
			if self.use_highway_connections:
				self.ht +=  (1 - rt) * xt
				
			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H

	def reset_state(self):
		self.set_state(None, None, None)

	def set_state(self, ct, ht, H):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.H = H		# all hidden states

	def get_last_hidden_state(self):
		return self.ht

	def get_all_hidden_states(self):
		return self.H
