import chainer, math
from chainer import cuda, link, functions, initializers
from chainer.links import ConvolutionND, Bias

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
			ft = functions.sigmoid(self.bf(F[..., t]))
			rt = functions.sigmoid(self.br(F[..., t]))

			self.ct = 0 if self.C is None else self.ct
			self.ct = ft * self.ct + (1 - ft) * zt

			if self.C is None:
				self.C = functions.expand_dims(self.ct, 2)
			else:
				self.C = functions.concat((self.C, functions.expand_dims(self.ct, 2)), axis=2)

			g_ct = self.ct
			if self.use_tanh:
				g_ct = functions.tanh(self.ct)

			self.ht = rt * g_ct
			if self.use_highway_connections:
				self.ht += (1 - rt) * xt

			if self.H is None:
				self.H = functions.expand_dims(self.ht, 2)
			else:
				self.H = functions.concat((self.H, functions.expand_dims(self.ht, 2)), axis=2)

		return self.H, self.C

	def reset_state(self):
		self.set_state(None, None, None)

	def set_state(self, ct, ht, H):
		self.ct = ct	# last cell state
		self.ht = ht	# last hidden state
		self.C = C		# all cell states
		self.H = H		# all hidden states

	def get_last_hidden_state(self):
		return self.ht

	def get_all_hidden_states(self):
		return self.H
