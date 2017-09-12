from chainer import link
from chainer.functions.connection import convolution_nd

class SRUFunction(convolution_nd.ConvolutionND):
		
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(2 <= n_in, n_in <= 4)

		x_type = in_types[0]
		v_type = in_types[1]
		g_type = in_types[1]
		type_check.expect(
			x_type.dtype.kind == "f",
			v_type.dtype.kind == "f",
			g_type.dtype.kind == "f",
			x_type.ndim == self.ndim + 2,
			v_type.ndim == self.ndim + 2,
			g_type.ndim == self.ndim + 2,
			x_type.shape[1] == v_type.shape[1],
		)

		if type_check.eval(n_in) == 4:
			b_type = in_types[3]
			type_check.expect(
				b_type.dtype == x_type.dtype,
				b_type.ndim == 1,
				b_type.shape[0] == v_type.shape[0],
			)
			
	def forward(self, inputs):
		x, V, g = inputs[:3]
		b = inputs[3] if len(inputs) == 4 else None
		
		self.norm = _get_norm(V)
		self.V_normalized = V / self.norm
		self.W = g * self.V_normalized

		if b is None:
			return super(SRUFunction, self).forward((x, self.W))
		return super(SRUFunction, self).forward((x, self.W, b))

	def backward(self, inputs, grad_outputs):
		x, V, g = inputs[:3]
		if hasattr(self, "W") == False:
			self.norm = _get_norm(V)
			self.V_normalized = V / self.norm
			self.W = g * self.V_normalized

		b = inputs[3] if len(inputs) == 4 else None
		if b is None:
			gx, gW = super(SRUFunction, self).backward((x, self.W), grad_outputs)
		else:
			gx, gW, gb = super(SRUFunction, self).backward((x, self.W, b), grad_outputs)

		xp = cuda.get_array_module(x)
		gg = xp.sum(gW * self.V_normalized, axis=(1, 2), keepdims=True)
		gV = g * (gW - gg * self.V_normalized) / self.norm

		if b is None:
			return gx, gV, gg
		else:
			return gx, gV, gg, gb

def sru(x, W, br, bf, cover_all=False):
	func = SRUFunction(cover_all)
	return func(x, W, br, bf)

class SRU(link.Link):

	def __init__(self, in_channels, out_channels, initialW=None):
		super().__init__()
		seoflin_channels = in_channels
		self.out_channels = out_channels

		with self.init_scope():
			W_initializer = initializers._get_initializer(initialW)
			self.W = variable.Parameter(W_initializer)
			if in_channels is not None:
				kh, kw = _pair(self.ksize)
				W_shape = (self.out_channels, in_channels, kh, kw)
				self.W.initialize(W_shape)

			self.br = None if nobias else variable.Parameter(None)
			self.bf = None if nobias else variable.Parameter(None)

	def _initialize_params(self, t):
		xp = cuda.get_array_module(t)
		# 出力チャネルごとにミニバッチ平均をとる
		mean_t = xp.mean(t, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
		std_t = xp.sqrt(xp.var(t, axis=(0, 2, 3))).reshape(1, -1, 1, 1)
		g = 1 / std_t
		b = -mean_t / std_t

		# print "g <- {}, b <- {}".format(g.reshape((-1,)), b.reshape((-1,)))

		with self.init_scope():
			if self.nobias == False:
				self.b = variable.Parameter(b.reshape((-1,)))
			self.g = variable.Parameter(g.reshape((self.out_channels, 1, 1, 1)))

		return mean_t, std_t

	@property
	def W(self):
		W = self.W.data
		xp = cuda.get_array_module(W)
		norm = _norm(W)
		W = W / norm
		return self.g.data * W

	def __call__(self, x):
		return sru(x, self.W, self.g, self.b, self.stride, self.pad)
