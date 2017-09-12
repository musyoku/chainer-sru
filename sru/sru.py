import cupy
from chainer import link, initializers, variable, cuda
from chainer.functions.connection import convolution_nd
from chainer.utils import conv_nd, type_check
from cupy.cuda import function
from pynvrtc.compiler import Program
# from pynvrtc.interface import NVRTCInterface, NVRTCException

CUDA_SRU_KERNEL = """
extern "C" 
{
	__forceinline__ __device__ 
	float sigmoidf(float x)
	{
		return 1.f / (1.f + expf(-x));
	}

	__global__ 
	void forward(const float* __restrict__ u)
	{
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		*(u + col) = (float)col;
	}
}
"""

# hack
tmp = cupy.random.uniform()

prog = Program(CUDA_SRU_KERNEL.encode("utf-8"), "sru.cu".encode("utf-8"))
cuda_module = function.Module()
cuda_module.load(bytes(prog.compile().encode()))
cuda_forward_func = cuda_module.get_function("forward")

# interface = NVRTCInterface()
# prog = interface.nvrtcCreateProgram(CUDA_SRU_KERNEL.encode("utf-8"), "sru.cu".encode("utf-8"), [], []);
# interface.nvrtcCompileProgram(prog, ["-ftz=true".encode("utf-8")])
# ptx = interface.nvrtcGetPTX(prog)
# module = function.Module()
# module.load(bytes(ptx.encode()))
# cuda_forward_func = module.get_function('forward')

class SRUFunction(convolution_nd.ConvolutionND):
	def __init__(self, cover_all=False):
		super().__init__(1, 1, 0, cover_all)
		
	def check_type_forward(self, in_types):
		n_in = in_types.size()
		type_check.expect(n_in == 3)

		x_type = in_types[0]
		w_type = in_types[1]
		b_type = in_types[2]
		type_check.expect(
			x_type.dtype.kind == "f",
			w_type.dtype.kind == "f",
			b_type.dtype.kind == "f",
			x_type.ndim == self.ndim + 2,
			w_type.ndim == self.ndim + 2,
			b_type.ndim == 1,
			b_type.shape[0] * 3 == w_type.shape[0] * 2,
		)
			
	def forward(self, inputs):
		x, W, b = inputs
		WX = super().forward((x, W))[0]
		xp = cuda.get_array_module(WX)
		batchsize = x.shape[0]
		print(WX.shape)
		H = xp.empty((batchsize, WX.shape[1], WX.shape[2]), dtype=x.dtype)
		cuda_forward_func(args=[
			H.data_ptr()
		])
		print(H)
		raise Exception()
		return WX,

	def backward(self, inputs, grad_outputs):
		raise NotImplementedError()

def sru(x, W, b, cover_all=False):
	func = SRUFunction(cover_all)
	return func(x, W, b)

class SRU(link.Link):

	def __init__(self, in_channels, out_channels, initialW=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		ksize = conv_nd.as_tuple(1, 1)

		with self.init_scope():
			self.W = variable.Parameter(initializers._get_initializer(initialW), (out_channels * 3, in_channels) + ksize)
			self.b = variable.Parameter(initializers._get_initializer(0), out_channels * 2)

	def __call__(self, x):
		return sru(x, self.W, self.b)
