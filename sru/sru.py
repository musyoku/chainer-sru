import cupy
import numpy as np
from chainer import link, initializers, variable, cuda, Function
from chainer.utils import conv_nd, type_check
from cupy.cuda import compiler, function

CUDA_SRU_KERNEL = """
extern "C" 
{
	__forceinline__ __device__ 
	float sigmoidf(float x)
	{
		return 1.f / (1.f + expf(-x));
	}

	__global__ 
	void forward(const float* __restrict__ x_ptr, const float* __restrict__ u_ptr, const float* __restrict__ bias_ptr, 
			float* __restrict__ context_ptr, float* __restrict__ hidden_state_ptr, 
			const int batchsize, const int feature_dimension, const int seq_length, const int use_tanh)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;
		int total_columns = batchsize * feature_dimension;
		if(column >= total_columns) return;
		int batch_index = column / feature_dimension;

		const float bf = *(bias_ptr + column % feature_dimension);
		const float br = *(bias_ptr + column % feature_dimension + feature_dimension);

		float* ct_ptr = context_ptr + column;
		float* ht_ptr = hidden_state_ptr + column * seq_length;
		const float* xt_ptr = x_ptr + column * seq_length;

		float ct = *(ct_ptr);

		const float* wr_ptr = u_ptr + column % feature_dimension * seq_length + batch_index * 3 * feature_dimension * seq_length;
		const float* wf_ptr = u_ptr + column % feature_dimension * seq_length + (batch_index * 3 + 1) * feature_dimension * seq_length;
		const float* z_ptr  = u_ptr + column % feature_dimension * seq_length + (batch_index * 3 + 2) * feature_dimension * seq_length;

		for(int t = 0;t < seq_length;t++)
		{
			float zt = *(z_ptr);					// x_tilde
			float ft = sigmoidf((*(wf_ptr)) + bf);
			float rt = sigmoidf((*(wr_ptr)) + br);
			float xt = *xt_ptr;

			ct = ft * (ct - zt) + zt;
			*ct_ptr = ct;
			
			ct = use_tanh ? tanh(ct) : ct;
			*ht_ptr = rt * (ct - xt) + xt;

			ht_ptr += 1;
			xt_ptr += 1;
			wr_ptr += 1;
			wf_ptr += 1;
			z_ptr += 1;
		}
	}

	__global__ 
	void backward(const float* __restrict__ x_ptr, const float* __restrict__ u_ptr, const float* __restrict__ bias_ptr, 
			float* __restrict__ context_ptr, float* __restrict__ hidden_state_ptr, 
			const int batchsize, const int feature_dimension, const int seq_length, const int use_tanh)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;
		int total_columns = batchsize * feature_dimension;
		if(column >= total_columns) return;
		int batch_index = column / feature_dimension;

		const float bf = *(bias_ptr + column % feature_dimension);
		const float br = *(bias_ptr + column % feature_dimension + feature_dimension);

		float* ct_ptr = context_ptr + column;
		float* ht_ptr = hidden_state_ptr + column * seq_length;
		const float* xt_ptr = x_ptr + column * seq_length;

		float ct = *(ct_ptr);

		const float* wr_ptr = u_ptr + column % feature_dimension * seq_length + batch_index * 3 * feature_dimension * seq_length;
		const float* wf_ptr = u_ptr + column % feature_dimension * seq_length + (batch_index * 3 + 1) * feature_dimension * seq_length;
		const float* z_ptr  = u_ptr + column % feature_dimension * seq_length + (batch_index * 3 + 2) * feature_dimension * seq_length;

		for(int t = 0;t < seq_length;t++)
		{
			float zt = *(z_ptr);
			float ft = sigmoidf((*(wf_ptr)) + bf);
			float rt = sigmoidf((*(wr_ptr)) + br);
			float xt = *xt_ptr;

			ct = ft * (ct - zt) + zt;
			ct = use_tanh ? tanh(ct) : ct;
			*ct_ptr = ct;
			*ht_ptr = rt * (ct - xt) + xt;

			ht_ptr += 1;
			xt_ptr += 1;
			wr_ptr += 1;
			wf_ptr += 1;
			z_ptr += 1;
		}
	}
}
"""

options = ["-ftz=true"]
nvcc = None
cupy_version = 0
if hasattr(compiler, "compile_using_nvrtc"):	# CuPy v2
	nvcc = compiler.compile_using_nvrtc
	cupy_version = 2
elif hasattr(compiler, "nvcc"):					# CuPy v1
	nvcc = compiler.nvcc
	cupy_version = 1
else:
	raise NotImplementedError()
ptx = nvcc(CUDA_SRU_KERNEL, options, None)

def _cuda_get_module():
	module = function.Module()
	
	if cupy_version == 1:
		module.load(ptx)
		return module

	if cupy_version == 2:
		ls = function.LinkState()
		ls.add_ptr_data(ptx, u"cupy.ptx")
		module.load(ls.complete())
		return module

	raise NotImplementedError()

def _cuda_elementwise(name, args, block, grid):
	module = _cuda_get_module()
	func = module.get_function(name)
	func(args=args, block=block, grid=grid)

def _np_sigmoid(x):
	return 1 / (1 + np.exp(-x))

class SRUFunction(Function):

	def __init__(self, use_tanh):
		self.use_tanh = use_tanh

	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(3 <= n_in, n_in <= 4)

		x_type = in_types[0]
		w_type = in_types[1]
		b_type = in_types[2]
		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			b_type.dtype.kind == "f",
			x_type.ndim == 3,
			w_type.ndim == 2,
			b_type.ndim == 1,
			b_type.shape[0] * 3 == w_type.shape[0] * 2,
		)

		if type_check.eval(n_in) == 4:
			ct_type = in_types[3]
			type_check.expect(
				ct_type.dtype == x_type.dtype,
				ct_type.ndim == 2,
				ct_type.shape[0] == b_type.shape[0],
			)

	# x: (batchsize, seq_length, feature_dimension)
	def forward_cpu(self, inputs):
		X, W, b = inputs[:3]
		batchsize, feature_dimension, seq_length = X.shape
		ct = inputs[3] if len(inputs) == 4 else np.zeros((batchsize, feature_dimension), dtype=X.dtype)

		U = np.matmul(W, X)
		R, F, Z = np.split(U, 3, axis=1)
		H = None

		for t in range(seq_length):
			xt = X[..., t]
			zt = Z[..., t]
			ft = _np_sigmoid(F[..., t] + b[:feature_dimension])
			rt = _np_sigmoid(R[..., t] + b[feature_dimension:])

			ct = ft * ct + (1 - ft) * zt

			g_ct = ct
			if self.use_tanh:
				g_ct = np.tanh(ct)

			ht = rt * g_ct + (1 - rt) * xt

			if H is None:
				H = np.expand_dims(ht, 2)
			else:
				H = np.concatenate((H, np.expand_dims(ht, 2)), axis=2)

		return H,

	def forward_gpu(self, inputs):
		X, W, b = inputs[:3]
		xp = cuda.get_array_module(W)
		batchsize, feature_dimension, seq_length = X.shape
		ct = inputs[3] if len(inputs) == 4 else xp.zeros((batchsize, feature_dimension), dtype=X.dtype)

		U = xp.matmul(W, X)
		# print(U.shape)
		# print(U)
		# ct += xp.random.uniform(size=(batchsize, feature_dimension))
		# ct = X[..., 0]
		# print(ct.data.ptr)
		# print(ct)

		total_columns = feature_dimension * batchsize
		thread_per_block = min(512, total_columns)
		num_block = total_columns // thread_per_block + 1

		H = xp.zeros((batchsize, feature_dimension, seq_length), dtype=X.dtype)
		# print(X.shape)
		# print(U.shape)
		_cuda_elementwise("forward", 
			args=[
				X.data.ptr,
				U.data.ptr,
				b.data.ptr,
				ct.data.ptr,
				H.data.ptr,
				batchsize,
				feature_dimension,
				seq_length,
				self.use_tanh
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))
		# import numpy
		# numpy.set_printoptions(suppress=True)
		# print(ct)
		# print(cuda.to_cpu(H).astype(numpy.float32))
		return H,

	def backward_cpu(self, inputs, grad_outputs):
		raise NotImplementedError()

	def backward_gpu(self, inputs, grad_outputs):
		raise NotImplementedError()

def sru(x, U, b, ct=None, use_tanh=True):
	func = SRUFunction(use_tanh)
	if ct is None:
		return func(x, U, b)
	return func(x, U, b, ct)

class SRU(link.Link):
	def __init__(self, in_channels, out_channels, use_tanh=True, initialW=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.use_tanh = use_tanh
		self.ct = None

		with self.init_scope():
			self.U = variable.Parameter(initializers._get_initializer(initialW), (out_channels * 3, in_channels))
			self.b = variable.Parameter(initializers._get_initializer(0), out_channels * 2)

	def __call__(self, x):
		return sru(x, self.U, self.b, self.ct, self.use_tanh)

	def reset_state(self):
		self.ct = None