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
		return 1.0f / (1.0f + expf(-x));
	}

	__global__ 
	void forward(const float* __restrict__ x_ptr, 
				 const float* __restrict__ u_ptr, 
				 const float* __restrict__ bias_ptr, 
				 const float* __restrict__ prev_cell_ptr, 
				 float* __restrict__ cell_ptr, 
				 float* __restrict__ hidden_state_ptr, 
				 const int batchsize, 
				 const int feature_dimension, 
				 const int seq_length, 
				 const int use_tanh)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;			// 0 <= column < batchsize * feature_dimension
		int total_columns = batchsize * feature_dimension;
		if(column >= total_columns) return;

		int batch_index = column / feature_dimension;				// 0 <= batch_index < batchsize
		int feature_index = column % feature_dimension;				// 0 <= feature_index < feature_dimension

		// B = (b_f, b_r)
		const float bf = *(bias_ptr + feature_index);
		const float br = *(bias_ptr + feature_index + feature_dimension);

		const float* prev_ct_ptr = prev_cell_ptr + column;			// c_{t-1}
		float* ct_ptr = cell_ptr + column * seq_length;				// c_t
		float* ht_ptr = hidden_state_ptr + column * seq_length;		// h_t
		const float* xt_ptr = x_ptr + column * seq_length;			// x_t

		float ct = *(prev_ct_ptr);	// initialize c_t

		// U = (W_r, W_f, W)
		const float* wrt_ptr = u_ptr + (feature_index + (batch_index * 3)     * feature_dimension) * seq_length;
		const float* wft_ptr = u_ptr + (feature_index + (batch_index * 3 + 1) * feature_dimension) * seq_length;
		const float*  zt_ptr = u_ptr + (feature_index + (batch_index * 3 + 2) * feature_dimension) * seq_length;

		for(int t = 0;t < seq_length;t++)
		{
			const float zt = *(zt_ptr);					// x_tilde_t
			const float ft = sigmoidf((*(wft_ptr)) + bf);
			const float rt = sigmoidf((*(wrt_ptr)) + br);
			const float xt = *xt_ptr;

			ct = ft * (ct - zt) + zt;
			*ct_ptr = ct;
			
			float g_ct = use_tanh ? tanh(ct) : ct;
			*ht_ptr = rt * (g_ct - xt) + xt;

			// move to the next time
			ht_ptr += 1;
			ct_ptr += 1;
			xt_ptr += 1;
			wrt_ptr += 1;
			wft_ptr += 1;
			zt_ptr  += 1;
		}
	}

	__global__ 
	void backward(const float* __restrict__ x_ptr, 
				  const float* __restrict__ u_ptr, 
				  const float* __restrict__ bias_ptr, 
				  const float* __restrict__ prev_cell_ptr, 
				  const float* __restrict__ grad_y_ptr,
				  float* __restrict__ grad_x_ptr, 
				  float* __restrict__ grad_w_ptr,
				  float* __restrict__ grad_bias_ptr, 
				  float* __restrict__ grad_prev_ct_ptr,
				  const int batchsize, 
				  const int feature_dimension, 
				  const int seq_length, 
				  const int use_tanh)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;			// 0 <= column < batchsize * feature_dimension
		int total_columns = batchsize * feature_dimension;
		if(column >= total_columns) return;

		int batch_index = column / feature_dimension;				// 0 <= batch_index < batchsize
		int feature_index = column % feature_dimension;				// 0 <= feature_index < feature_dimension

		// B = (b_f, b_r)
		const float bf = *(bias_ptr + feature_index);
		const float br = *(bias_ptr + feature_index + feature_dimension);

		const float* prev_ct_ptr = prev_cell_ptr + column;			// c_{t-1}
		const float* xt_ptr = x_ptr + column * seq_length;			// x_t

		// U = (W_r, W_f, W)
		const float* wrt_ptr = u_ptr + (feature_index + (batch_index * 3)     * feature_dimension) * seq_length;
		const float* wft_ptr = u_ptr + (feature_index + (batch_index * 3 + 1) * feature_dimension) * seq_length;
		const float*  zt_ptr = u_ptr + (feature_index + (batch_index * 3 + 2) * feature_dimension) * seq_length;

		const float* grad_ht_ptr = grad_y_ptr + column * seq_length;	// gradient from the upper layer
		float prev_ct = *(prev_ct_ptr);	// initialize c_t

		// backward
		float* grad_bft_ptr = grad_bias_ptr + (feature_index + (batch_index * 2)     * feature_dimension) * seq_length;
		float* grad_brt_ptr = grad_bias_ptr + (feature_index + (batch_index * 2 + 1) * feature_dimension) * seq_length;
		float* grad_xt_ptr = grad_x_ptr + column * seq_length;

		// init
		*grad_bft_ptr = 0;
		float prev_grad_bft = 0;

		for(int t = 0;t < seq_length;t++)
		{
			// forward
			const float zt = *(zt_ptr);						// x_tilde_t
			const float ft = sigmoidf((*(wft_ptr)) + bf);
			const float rt = sigmoidf((*(wrt_ptr)) + br);
			const float xt = *xt_ptr;
			const float grad_y = *grad_ht_ptr;				// gradient from the upper layer

			float next_ct = ft * (prev_ct - zt) + zt;
			float g_ct = use_tanh ? tanh(next_ct) : next_ct;

			// backward
			*grad_brt_ptr = grad_y * (g_ct - xt) * (1.0f - rt) * rt;
			const float grad_tanh = use_tanh ? (1.0f - g_ct * g_ct) : 1.0f;
			*grad_bft_ptr = grad_y * rt * grad_tanh * (prev_ct - zt) * (1.0f - ft) * ft + grad_y * rt * ft * prev_grad_bft;
			*grad_xt_ptr = zt;

			prev_grad_bft = grad_y * rt * grad_tanh * (prev_ct - zt) * (1.0f - ft) * ft;

			// move to the next time
			xt_ptr  += 1;
			wrt_ptr += 1;
			wft_ptr += 1;
			zt_ptr  += 1;
			grad_ht_ptr  += 1;
			grad_xt_ptr  += 1;
			grad_brt_ptr += 1;
			grad_bft_ptr += 1;

			// update cell
			prev_ct = next_ct;
		}
	}

	__global__ 
	void backward_test(
				  float* __restrict__ grad_y_ptr,
				  float* __restrict__ grad_x_ptr, 
				  const int batchsize, 
				  const int feature_dimension, 
				  const int seq_length, 
				  const int use_tanh)
	{
		int column = blockIdx.x * blockDim.x + threadIdx.x;			// 0 <= column < batchsize * feature_dimension
		int total_columns = batchsize * feature_dimension;
		if(column >= total_columns) return;

		int batch_index = column / feature_dimension;				// 0 <= batch_index < batchsize
		int feature_index = column % feature_dimension;				// 0 <= feature_index < feature_dimension

		// backward
		float* grad_ht_ptr = grad_y_ptr + column * seq_length;	// gradient from the upper layer
		float* grad_xt_ptr = grad_x_ptr + column * seq_length;

		for(int t = 0;t < seq_length;t++)
		{
			const float grad_y = *grad_ht_ptr;				// gradient from the upper layer
			*grad_xt_ptr = column + t;
			*grad_ht_ptr = column + t;
			grad_ht_ptr += 1;
			grad_xt_ptr += 1;
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
		C = None

		for t in range(seq_length):
			xt = X[..., t]
			zt = Z[..., t]
			ft = _np_sigmoid(F[..., t] + b[:feature_dimension])
			rt = _np_sigmoid(R[..., t] + b[feature_dimension:])

			ct = ft * ct + (1 - ft) * zt

			if C is None:
				C = np.expand_dims(ct, 2)
			else:
				C = np.concatenate((C, np.expand_dims(ct, 2)), axis=2)

			g_ct = ct
			if self.use_tanh:
				g_ct = np.tanh(ct)

			ht = rt * g_ct + (1 - rt) * xt

			if H is None:
				H = np.expand_dims(ht, 2)
			else:
				H = np.concatenate((H, np.expand_dims(ht, 2)), axis=2)

		return H, C

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

		H = xp.empty((batchsize, feature_dimension, seq_length), dtype=X.dtype)
		C = xp.empty((batchsize, feature_dimension, seq_length), dtype=X.dtype)
		# print(X.shape)
		# print(U.shape)
		_cuda_elementwise("forward", 
			args=[
				X.data.ptr,
				U.data.ptr,
				b.data.ptr,
				ct.data.ptr,
				C.data.ptr,
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
		return H, C

	def backward_cpu(self, inputs, grad_outputs):
		raise NotImplementedError()

	def backward_gpu(self, inputs, grad_outputs):
		X, W, b = inputs[:3]
		xp = cuda.get_array_module(W)
		batchsize, feature_dimension, seq_length = X.shape
		ct = inputs[3] if len(inputs) == 4 else xp.zeros((batchsize, feature_dimension), dtype=X.dtype)

		total_columns = feature_dimension * batchsize
		thread_per_block = min(512, total_columns)
		num_block = total_columns // thread_per_block + 1

		U = xp.matmul(W, X)

		grad_x = xp.zeros_like(X)
		grad_b = xp.empty((batchsize, feature_dimension * 2, seq_length), dtype=b.dtype)
		grad_w = xp.zeros_like(W)
		grad_prev_ct = xp.zeros_like(ct)

		grad_h = grad_outputs[0]
		# print(grad_h.flags)
		grad_h = cupy.ascontiguousarray(grad_h)
		# print(grad_h.flags)

		_cuda_elementwise("backward", 
			args=[
				X.data.ptr,
				U.data.ptr,
				b.data.ptr,
				ct.data.ptr,
				grad_h.data.ptr,
				grad_x.data.ptr,
				grad_w.data.ptr,
				grad_b.data.ptr,
				grad_prev_ct.data.ptr,
				batchsize,
				feature_dimension,
				seq_length,
				self.use_tanh
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))
		# cuda.cupy.ElementwiseKernel(
		# 	'float32 x',
		# 	'float32 h',
		# 	'''
		# 	h = i;
		# 	''',
		# 	'reduce_probability')(grad_x, grad_h)

		# print("ElementwiseKernel")
		# print(grad_h)
		# print(grad_x)

		# _cuda_elementwise("backward_test", 
		# 	args=[
		# 		grad_h.data.ptr,
		# 		grad_x.data.ptr,
		# 		batchsize,
		# 		feature_dimension,
		# 		seq_length,
		# 		self.use_tanh
		# 	], 
		# 	block=(thread_per_block, 1, 1), 
		# 	grid=(num_block, 1, 1))



		# print("_cuda_elementwise")
		np.set_printoptions(suppress=True)
		print(grad_x)
		print(xp.sum(grad_b, axis=(0,)))
		grad_b = xp.sum(grad_b, axis=(0, 2))
		print(grad_b)






		return None, None, None

def sru(x, W, b, ct=None, use_tanh=True):
	func = SRUFunction(use_tanh)
	if ct is None:
		return func(x, W, b)
	return func(x, W, b, ct)

class SRU(link.Link):
	def __init__(self, in_channels, out_channels, use_tanh=True, initialW=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.use_tanh = use_tanh
		self.ct = None

		with self.init_scope():
			self.W = variable.Parameter(initializers._get_initializer(initialW), (out_channels * 3, in_channels))
			self.b = variable.Parameter(initializers._get_initializer(0), out_channels * 2)

	def __call__(self, x):
		return sru(x, self.W, self.b, self.ct, self.use_tanh)

	def reset_state(self):
		self.ct = None