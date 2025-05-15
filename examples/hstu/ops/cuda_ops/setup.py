import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]

nvcc_flags = [
    "-g",
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]
cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
setup(
    name='jagged_tensor_op',
    author='Runchu Zhao',
    description='JaggedTensor concat forward and backward',
    ext_modules=[
        CUDAExtension(
            name='jagged_tensor_op',
            sources=['csrc/jagged_tensor_op_cuda.cpp', 'csrc/jagged_tensor_op_kernel.cu'],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                # "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag,
                "nvcc": nvcc_threads_args() + nvcc_flags,
                # "nvcc": ["-O2"],
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)