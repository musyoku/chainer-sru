import cupy
from chainer import link, initializers, variable, cuda
from chainer.functions.connection import convolution_nd
from chainer.utils import conv_nd, type_check
from cupy.cuda import function
from pynvrtc.interface import NVRTCInterface, NVRTCException

CUDA_SRU_KERNEL = """
extern "C" 
{
	__forceinline__ __device__ 
	float sigmoidf(float x)
	{
		return 1.f / (1.f + expf(-x));
	}

	__global__ 
	void forward(const float* __restrict__ u, const float* __restrict__ x,
		const float* __restrict__ bias, const float* __restrict__ init,
		const float* __restrict__ mask_h,
		const int len, const int batch, const int d, const int k,
		float* __restrict__ h, float* __restrict__ c,
		const int use_tanh)
	{
		assert ((k == 3) || (x == NULL));

		int ncols = batch*d;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		if (col >= ncols) return;

		int ncols_u = ncols*k;
		int ncols_x = (k == 3) ? ncols : ncols_u;

		const float bias1 = *(bias + (col%d));
		const float bias2 = *(bias + (col%d) + d);
		const float mask = (mask_h == NULL) ? 1.0 : (*(mask_h + col));
		float cur = *(init + col);

		const float* up = u + (col*k);
		const float* xp = (k == 3) ? (x + col) : (up + 3);
		float* cp = c + col;
		float* hp = h + col;

		for (int row = 0; row < len; ++row)
		{
			float g1 = sigmoidf((*(up+1))+bias1);
			float g2 = sigmoidf((*(up+2))+bias2);
			cur = (cur-(*up))*g1 + (*up);
			*cp = cur;
			float val = use_tanh ? tanh(cur) : cur;
			*hp = (val*mask-(*xp))*g2 + (*xp);
			up += ncols_u;
			xp += ncols_x;
			cp += ncols;
			hp += ncols;
		}
	}
}
"""

# hack
# tmp = cupy.random.uniform()

interface = NVRTCInterface("/usr/local/cuda/lib64/libnvrtc.so")

try:
	prog = interface.nvrtcCreateProgram(CUDA_SRU_KERNEL.encode("utf-8"), "sru.cu".encode("utf-8"), [], []);
	interface.nvrtcCompileProgram(prog, ["-ftz=true".encode("utf-8")])
	ptx = interface.nvrtcGetPTX(prog)
	module = function.Module()
	module.load(bytes(ptx.encode()))
	SRU_FWD_FUNC = SRU_MOD.get_function('sru_forward')
	SRU_BWD_FUNC = SRU_MOD.get_function('sru_bwd')
	SRU_BiFWD_FUNC = SRU_MOD.get_function('sru_bi_fwd')
	SRU_BiBWD_FUNC = SRU_MOD.get_function('sru_bi_bwd')
except NVRTCException as e:
	print("Error: %s" % repr(e))


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
