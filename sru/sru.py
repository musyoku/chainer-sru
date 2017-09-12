from chainer import link, initializers, variable, cuda
from chainer.functions.connection import convolution_nd
from chainer.utils import conv_nd, type_check

class SRUFunction(convolution_nd.ConvolutionND):
	def __init__(self, cover_all=False):
		super().__init__(1, 1, 0, cover_all)
		
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 4)

		x_type = in_types[0]
		w_type = in_types[1]
		br_type = in_types[2]
		bf_type = in_types[3]
		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			br_type.dtype.kind == "f",
			bf_type.dtype.kind == "f",
			x_type.ndim == self.ndim + 2,
			w_type.ndim == self.ndim + 2,
			bf_type.ndim == 1,
			br_type.ndim == 1,
			bf_type.shape[0] * 3 == w_type.shape[0],
			br_type.shape[0] * 3 == w_type.shape[0],
		)
			
	def forward(self, inputs):
		x, W, br, bf = inputs
		WX = super().forward((x, W))[0]
		xp = cuda.get_array_module(WX)
		batchsize = x.shape[0]
		print(WX.shape)
		H = xp.empty((batchsize, WX.shape[1], WX.shape[2]), dtype=x.dtype)
		return WX

	def backward(self, inputs, grad_outputs):
		raise NotImplementedError()

def sru(x, W, br, bf, cover_all=False):
	func = SRUFunction(cover_all)
	return func(x, W, br, bf)

class SRU(link.Link):

	def __init__(self, in_channels, out_channels, initialW=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		ksize = conv_nd.as_tuple(1, 1)

		with self.init_scope():
			self.W = variable.Parameter(initializers._get_initializer(initialW), (out_channels * 3, in_channels) + ksize)
			self.br = variable.Parameter(initializers._get_initializer(0), out_channels)
			self.bf = variable.Parameter(initializers._get_initializer(0), out_channels)

	def __call__(self, x):
		return sru(x, self.W, self.br, self.bf)
