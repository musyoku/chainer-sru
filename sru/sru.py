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
		//return 1.0f / (1.0f + expf(-x));
		return tanh(x * 0.5f) * 0.5f + 0.5f;
	}

	__global__ 
	void forward(const float* __restrict__ x_ptr, 
				 const float* __restrict__ u_ptr, 
				 const float* __restrict__ bias_ptr, 
				 const float* __restrict__ initial_cell_ptr, 
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

		const float* initial_ct_ptr = initial_cell_ptr + column;			// initial cell state
		float* ct_ptr = cell_ptr + column * seq_length;				// c_t
		float* ht_ptr = hidden_state_ptr + column * seq_length;		// h_t
		const float* xt_ptr = x_ptr + column * seq_length;			// x_t

		float ct = *(initial_ct_ptr);	// initialize c_t

		// U = (W_r, W_f, W_z) @ X
		const float* uzt_ptr = u_ptr + (feature_index + (batch_index * 3)     * feature_dimension) * seq_length;
		const float* uft_ptr = u_ptr + (feature_index + (batch_index * 3 + 1) * feature_dimension) * seq_length;
		const float* urt_ptr = u_ptr + (feature_index + (batch_index * 3 + 2) * feature_dimension) * seq_length;

		for(int t = 0;t < seq_length;t++)
		{
			const float zt = *(uzt_ptr);					// x_tilde_t
			const float ft = sigmoidf((*(uft_ptr)) + bf);
			const float rt = sigmoidf((*(urt_ptr)) + br);
			const float xt = *xt_ptr;

			ct = ft * (ct - zt) + zt;
			*ct_ptr = ct;
			
			float g_ct = use_tanh ? tanh(ct) : ct;
			*ht_ptr = rt * (g_ct - xt) + xt;

			// move to the next time
			ht_ptr += 1;
			ct_ptr += 1;
			xt_ptr += 1;
			uzt_ptr += 1;
			uft_ptr += 1;
			urt_ptr += 1;
		}
	}

	__global__ 
	void backward(const float* __restrict__ x_ptr, 
				  const float* __restrict__ u_ptr, 
				  const float* __restrict__ bias_ptr, 
				  const float* __restrict__ cell_ptr, 
				  const float* __restrict__ initial_cell_ptr, 
				  const float* __restrict__ incoming_grad_h_ptr,
				  const float* __restrict__ incoming_grad_ct_ptr,
				  const float* __restrict__ w_ptr,
				  float* __restrict__ grad_highway_x_ptr, 
				  float* __restrict__ grad_u_ptr, 
				  float* __restrict__ grad_bias_ptr, 
				  float* __restrict__ grad_init_ct_ptr,
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

		const float* initial_ct_ptr = initial_cell_ptr + column;			// initial cell state
		const float* xt_ptr = x_ptr + column * seq_length;			// x_t
		const float* ct_ptr = cell_ptr + column * seq_length;	// c_t

		// U = (W_r, W_f, W_z) @ X
		const float* uzt_ptr = u_ptr + (feature_index + (batch_index * 3)     * feature_dimension) * seq_length;
		const float* uft_ptr = u_ptr + (feature_index + (batch_index * 3 + 1) * feature_dimension) * seq_length;
		const float* urt_ptr = u_ptr + (feature_index + (batch_index * 3 + 2) * feature_dimension) * seq_length;

		const float* incoming_grad_ht_ptr = incoming_grad_h_ptr + column * seq_length;	// gradient from the upper layer
		const float initial_cell = *(initial_ct_ptr);	// initialize c_t

		// gradient
		//// B
		float* grad_bft_ptr = grad_bias_ptr + (feature_index + (batch_index * 2)     * feature_dimension) * seq_length;
		float* grad_brt_ptr = grad_bias_ptr + (feature_index + (batch_index * 2 + 1) * feature_dimension) * seq_length;
		//// X
		float* grad_highway_xt_ptr = grad_highway_x_ptr + column * seq_length;
		//// U
		float* grad_uzt_ptr = grad_u_ptr + (feature_index + (batch_index * 3)     * feature_dimension) * seq_length;
		float* grad_uft_ptr = grad_u_ptr + (feature_index + (batch_index * 3 + 1) * feature_dimension) * seq_length;
		float* grad_urt_ptr = grad_u_ptr + (feature_index + (batch_index * 3 + 2) * feature_dimension) * seq_length;

		// move to time T
		xt_ptr  += seq_length - 1;
		urt_ptr += seq_length - 1;
		uft_ptr += seq_length - 1;
		uzt_ptr += seq_length - 1;
		ct_ptr  += seq_length - 1;
		grad_highway_xt_ptr += seq_length - 1;
		grad_brt_ptr += seq_length - 1;
		grad_bft_ptr += seq_length - 1;
		grad_uzt_ptr += seq_length - 1;
		incoming_grad_ht_ptr += seq_length - 1;

		float incoming_grad_ct = *(incoming_grad_ct_ptr + column);	// gradient propagating from time t to t-1

		for(int t = seq_length - 1;t >= 0;t--)
		{
			// forward
			const float zt = *(uzt_ptr);						// x_tilde_t
			const float ft = sigmoidf((*(uft_ptr)) + bf);
			const float rt = sigmoidf((*(urt_ptr)) + br);
			const float xt = *xt_ptr;
			const float incoming_grad_ht = *incoming_grad_ht_ptr;	// gradient from the upper layer
			const float ct = *(ct_ptr);						// c_t
			const float prev_ct = t == 0 ? initial_cell : *(ct_ptr - 1);	// c_{t-1}

			float g_ct = use_tanh ? tanh(ct) : ct;

			// backward
			//// b_r
			*grad_brt_ptr = incoming_grad_ht * (g_ct - xt) * (1.0f - rt) * rt;

			//// b_f
			const float grad_tanh = use_tanh ? (1.0f - g_ct * g_ct) : 1.0f;
			const float grad_ct = incoming_grad_ht * rt * grad_tanh;
			*grad_bft_ptr = (grad_ct + incoming_grad_ct) * (prev_ct - zt) * (1 - ft) * ft;

			//// x_t (highway connection)
			*grad_highway_xt_ptr = incoming_grad_ht * (1.0f - rt);

			//// U_t
			*grad_uzt_ptr = (incoming_grad_ht * rt * grad_tanh + incoming_grad_ct) * (1.0f - ft);
			*grad_uft_ptr = *grad_bft_ptr;
			*grad_urt_ptr = *grad_brt_ptr;

			//// c_{t-1}
			incoming_grad_ct = (grad_ct + incoming_grad_ct) * ft;

			// move to the prev time
			xt_ptr  -= 1;
			urt_ptr -= 1;
			uft_ptr -= 1;
			uzt_ptr -= 1;
			ct_ptr  -= 1;
			incoming_grad_ht_ptr -= 1;
			grad_highway_xt_ptr -= 1;
			grad_uzt_ptr -= 1;
			grad_uft_ptr -= 1;
			grad_urt_ptr -= 1;
			grad_brt_ptr -= 1;
			grad_bft_ptr -= 1;
		}
		*(grad_init_ct_ptr + column) = incoming_grad_ct;
	}

	__global__ 
	void backward_grad_x(
			const float* __restrict__ w_ptr, 
			float* __restrict__ grad_x_ptr, 
			const float* __restrict__ grad_highway_x_ptr, 
			const float* __restrict__ grad_u_ptr, 
			const int batchsize, 
			const int feature_dimension, 
			const int seq_length)
	{
		int identifier = blockIdx.x * blockDim.x + threadIdx.x;				// 0 <= identifier < batchsize * feature_dimension * seq_length
		int total_threads = batchsize * feature_dimension * seq_length;
		if(identifier >= total_threads) return;

		int t = identifier % seq_length;
		int batch_index = identifier / (feature_dimension * seq_length);	// 0 <= batch_index < batchsize
		int feature_index = (identifier / seq_length) % feature_dimension;					// 0 <= feature_index < feature_dimension

		// xp.dot(grad_u.transpose((0, 2, 1)), W).transpose((0, 2, 1))
		float* grad_xt_ptr = grad_x_ptr + identifier;
		*grad_xt_ptr = 0;
		for(int d = 0;d < feature_dimension * 3;d++)
		{
			const float* gptr = grad_u_ptr + (d + (batch_index * 3) * feature_dimension) * seq_length + t;
			*grad_xt_ptr += *(w_ptr + d * feature_dimension + feature_index) * *gptr;
		}
		// highway connection
		*grad_xt_ptr += *(grad_highway_x_ptr + identifier);
	}

	__global__ 
	void backward_grad_w(
			const float* __restrict__ grad_x_ptr, 
			const float* __restrict__ grad_u_ptr, 
			float* __restrict__ grad_w_ptr, 
			const int batchsize, 
			const int feature_dimension, 
			const int seq_length)
	{
		int identifier = blockIdx.x * blockDim.x + threadIdx.x;				// 0 <= identifier < batchsize * feature_dimension * seq_length
		int total_threads = feature_dimension * feature_dimension * 3;
		if(identifier >= total_threads) return;

		int batch_index = identifier / (feature_dimension * seq_length);	// 0 <= batch_index < batchsize
		int feature_index = (identifier / seq_length) % feature_dimension;					// 0 <= feature_index < feature_dimension

		*grad_w_ptr = feature_index;
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

CUDA_SRU_PTX = nvcc(CUDA_SRU_KERNEL, options, None)

def _np_sigmoid(x):
	return 1 / (1 + np.exp(-x))

def _as_contiguous(args):
	if isinstance(args, (list, tuple)):
		ret = []
		for arg in args:
			if arg is None:
				ret.append(None)
				continue
			if arg.flags.c_contiguous is False:
				arg = cupy.ascontiguousarray(arg)
			ret.append(arg)
		return ret

	if args.flags.c_contiguous is False:
		args = cupy.ascontiguousarray(args)

	return args

class SRUFunction(Function):

	_cuda_module = None

	def __init__(self, use_tanh):
		self.use_tanh = use_tanh

	def _cuda_elementwise(self, name, args, block, grid):
		module = self._cuda_get_module()
		func = module.get_function(name)
		func(args=args, block=block, grid=grid)

	def _cuda_get_module(self):
		if self._cuda_module is not None:
			return self._cuda_module

		self._cuda_module = function.Module()
		
		if cupy_version == 1:
			self._cuda_module.load(CUDA_SRU_PTX)
			return self._cuda_module

		if cupy_version == 2:
			ls = function.LinkState()
			ls.add_ptr_data(CUDA_SRU_PTX, u"cupy.ptx")
			self._cuda_module.load(ls.complete())
			return self._cuda_module

		raise NotImplementedError()

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
				ct_type.shape[1] == x_type.shape[1],
			)

	# x: (batchsize, feature_dimension, seq_length)
	def forward_cpu(self, inputs):
		X, W, B = inputs[:3]
		batchsize, feature_dimension, seq_length = X.shape
		ct = inputs[3] if len(inputs) == 4 else np.zeros((batchsize, feature_dimension), dtype=X.dtype)

		U = np.matmul(W, X)
		Z, F, R = np.split(U, 3, axis=1)
		H = None
		C = None

		for t in range(seq_length):
			xt = X[..., t]
			zt = Z[..., t]
			ft = _np_sigmoid(F[..., t] + B[:feature_dimension])
			rt = _np_sigmoid(R[..., t] + B[feature_dimension:])

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

		return H, C, C[..., -1]

	# x: (batchsize, feature_dimension, seq_length)
	def forward_gpu(self, inputs):
		X, W, B = _as_contiguous(inputs[:3])
		
		xp = cuda.get_array_module(W)
		batchsize, feature_dimension, seq_length = X.shape

		initial_ct = _as_contiguous(inputs[3]) if len(inputs) == 4 else xp.zeros((batchsize, feature_dimension), dtype=X.dtype)

		U = xp.matmul(W, X)

		total_columns = feature_dimension * batchsize
		thread_per_block = min(512, total_columns)
		num_block = total_columns // thread_per_block + 1

		H = xp.empty((batchsize, feature_dimension, seq_length), dtype=X.dtype)
		C = xp.empty((batchsize, feature_dimension, seq_length), dtype=X.dtype)
		
		self._cuda_elementwise("forward", 
			args=[
				X.data.ptr,
				U.data.ptr,
				B.data.ptr,
				initial_ct.data.ptr,
				C.data.ptr,
				H.data.ptr,
				batchsize,
				feature_dimension,
				seq_length,
				self.use_tanh
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))

		self.U = U
		self.C = C
		return H, C, C[..., -1]

	def backward_cpu(self, inputs, grad_outputs):
		raise NotImplementedError()

	def backward_gpu(self, inputs, grad_outputs):
		X, W, B = _as_contiguous(inputs[:3])
		xp = cuda.get_array_module(W)
		batchsize, feature_dimension, seq_length = X.shape
		initial_ct = _as_contiguous(inputs[3]) if len(inputs) == 4 else xp.zeros((batchsize, feature_dimension), dtype=X.dtype)


		total_columns = feature_dimension * batchsize
		thread_per_block = min(512, total_columns)
		num_block = total_columns // thread_per_block + 1

		grad_x = xp.zeros_like(X)
		grad_highway_x = xp.zeros_like(X)
		grad_b = xp.zeros((batchsize, feature_dimension * 2, seq_length), dtype=B.dtype)
		grad_w = xp.zeros((batchsize,) + W.shape + (seq_length,), dtype=W.dtype)
		grad_u = xp.zeros((batchsize, feature_dimension * 3, seq_length), dtype=self.U.dtype)
		grad_initial_ct = xp.zeros_like(initial_ct)

		# initialize with zero
		incoming_grad_ct = xp.zeros_like(initial_ct) if grad_outputs[2] is None else _as_contiguous(grad_outputs[2])
		incoming_grad_h = xp.zeros_like(X) if grad_outputs[0] is None else _as_contiguous(grad_outputs[0])

		self._cuda_elementwise("backward", 
			args=[
				X.data.ptr,
				self.U.data.ptr,
				B.data.ptr,
				self.C.data.ptr,
				initial_ct.data.ptr,
				incoming_grad_h.data.ptr,
				incoming_grad_ct.data.ptr,
				W.data.ptr,
				grad_highway_x.data.ptr,
				grad_u.data.ptr,
				grad_b.data.ptr,
				grad_initial_ct.data.ptr,
				batchsize,
				feature_dimension,
				seq_length,
				self.use_tanh
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))

		# grad_u = xp.concatenate((grad_uz, grad_b), axis=1)
		# grad_u = xp.concatenate((grad_u[:, :feature_dimension, :], grad_b), axis=1)

		# print(grad_x)
		# print(W)
		# print(xp.sum(W, axis=0))
		# print(xp.dot(grad_u.transpose((0, 2, 1)), W).transpose((0, 2, 1)) + grad_highway_x)
		# raise Exception()

		total_threads = feature_dimension * batchsize * seq_length
		thread_per_block = min(512, total_threads)
		num_block = total_threads // thread_per_block + 1

		self._cuda_elementwise("backward_grad_x", 
			args=[
				W.data.ptr,
				grad_x.data.ptr,
				grad_highway_x.data.ptr,
				grad_u.data.ptr,
				batchsize,
				feature_dimension,
				seq_length
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))

		# import numpy
		# numpy.set_printoptions(suppress=True)
		# _grad_x = xp.dot(grad_u.transpose((0, 2, 1)), W).transpose((0, 2, 1)) + grad_highway_x
		# print(xp.mean(abs(grad_x - _grad_x)))
		# # print(grad_x)
		# print(grad_b)
		grad_b = xp.sum(grad_b, axis=(0, 2))
		# print(grad_b)

		if len(inputs) == 4:
			return grad_x, W, grad_b, grad_initial_ct
		return grad_x, W, grad_b
		
		total_threads = feature_dimension ** 2 * 3 * batchsize
		thread_per_block = min(512, total_threads)
		num_block = total_threads // thread_per_block + 1

		self._cuda_elementwise("backward_grad_w", 
			args=[
				grad_x.data.ptr,
				grad_u.data.ptr,
				grad_w.data.ptr,
				batchsize,
				feature_dimension,
				seq_length
			], 
			block=(thread_per_block, 1, 1), 
			grid=(num_block, 1, 1))


		if len(inputs) == 4:
			return grad_x, W, grad_b, grad_initial_ct
		return grad_x, W, grad_b

		# raise Exception()

		grad_w = xp.broadcast_to(grad_u[..., None, :], (batchsize,) + W.shape + (seq_length,))
		grad_w = xp.sum(grad_w * X[:, None, ...], axis=(0, 3))


		if len(inputs) == 4:
			return grad_x, grad_w, grad_b, grad_initial_ct
		return grad_x, grad_w, grad_b

def sru(x, W, B, initial_ct=None, use_tanh=True):
	func = SRUFunction(use_tanh)
	if initial_ct is None:
		return func(x, W, B)
	return func(x, W, B, initial_ct)

class SRU(link.Link):
	def __init__(self, in_channels, out_channels, use_tanh=True, initialW=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.use_tanh = use_tanh

		with self.init_scope():
			self.W = variable.Parameter(initializers._get_initializer(initialW), (out_channels * 3, in_channels))
			self.B = variable.Parameter(initializers._get_initializer(0), out_channels * 2)

	def __call__(self, x, initial_ct):
		return sru(x, self.W, self.B, initial_ct, self.use_tanh)
