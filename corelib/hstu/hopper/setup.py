# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.

import copy
import itertools
import os
import platform
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "hstu-hopper"

# FORCE_BUILD: Force a fresh build locally, instead of attempting to find prebuilt wheels
# SKIP_CUDA_BUILD: Intended to allow CI to use a simple `python setup.py sdist` run to copy over raw files, without any cuda compilation
FORCE_BUILD = os.getenv("HSTU_FORCE_BUILD", "FALSE") == "TRUE"
SKIP_CUDA_BUILD = os.getenv("HSTU_SKIP_CUDA_BUILD", "FALSE") == "TRUE"
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("HSTU_FORCE_CXX11_ABI", "FALSE") == "TRUE"

DISABLE_BACKWARD = os.getenv("HSTU_DISABLE_BACKWARD", "FALSE") == "TRUE"
DISABLE_LOCAL = os.getenv("HSTU_DISABLE_LOCAL", "FALSE") == "TRUE"
DISABLE_CAUSAL = os.getenv("HSTU_DISABLE_CAUSAL", "FALSE") == "TRUE"
DISABLE_CONTEXT = os.getenv("HSTU_DISABLE_CONTEXT", "FALSE") == "TRUE"
DISABLE_TARGET = os.getenv("HSTU_DISABLE_TARGET", "FALSE") == "TRUE"
DISABLE_DELTA_Q = os.getenv("HSTU_DISABLE_DELTA_Q", "FALSE") == "TRUE"
DISABLE_RAB = os.getenv("HSTU_DISABLE_RAB", "FALSE") == "TRUE"
DISABLE_DRAB = os.getenv("HSTU_DISABLE_DRAB", "FALSE") == "TRUE"
DISABLE_BF16 = os.getenv("HSTU_DISABLE_BF16", "FALSE") == "TRUE"
DISABLE_FP16 = os.getenv("HSTU_DISABLE_FP16", "FALSE") == "TRUE"
DISABLE_FP8 = os.getenv("HSTU_DISABLE_FP8", "FALSE") == "TRUE"
DISABLE_HDIM32 = os.getenv("HSTU_DISABLE_HDIM32", "FALSE") == "TRUE"
DISABLE_HDIM64 = os.getenv("HSTU_DISABLE_HDIM64", "FALSE") == "TRUE"
DISABLE_HDIM128 = os.getenv("HSTU_DISABLE_HDIM128", "FALSE") == "TRUE"
DISABLE_HDIM256 = os.getenv("HSTU_DISABLE_HDIM256", "FALSE") == "TRUE"
DISABLE_SM8x = os.getenv("HSTU_DISABLE_SM8x", "FALSE") == "TRUE"

ONLY_COMPILE_SO = os.getenv("HSTU_ONLY_COMPILE_SO", "FALSE") == "TRUE"

if ONLY_COMPILE_SO:
    CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if CUDA_HOME is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))

    COMMON_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--compiler-options",
        "'-fPIC'",
    ]

    # a navie/minial way to build a cuda so, should only work under unix and may have differences compared to torch's BuildExtension
    def CUDAExtension(name, sources, *args, **kwargs):
        library_dirs = kwargs.get("library_dirs", [])
        library_dirs.append(os.path.join(CUDA_HOME, "lib64"))
        kwargs["library_dirs"] = library_dirs
        libraries = kwargs.get("libraries", [])
        kwargs["libraries"] = libraries

        include_dirs = kwargs.get("include_dirs", [])
        include_dirs.append(os.path.join(CUDA_HOME, "include"))
        kwargs["include_dirs"] = include_dirs

        kwargs["language"] = "c++"
        return Extension(name, sources, *args, **kwargs)

    class BuildExtension(build_ext):
        def build_extensions(self):
            self.compiler.src_extensions += [".cu", ".cuh"]
            original_compile = self.compiler._compile

            def unix_wrap_single_compile(
                obj, src, ext, cc_args, extra_postargs, pp_opts
            ) -> None:
                # Copy before we make any modifications.
                cflags = copy.deepcopy(extra_postargs)
                try:
                    original_compiler = self.compiler.compiler_so
                    if src.endswith(".cu"):
                        nvcc = [os.path.join(CUDA_HOME, "bin", "nvcc")]
                        self.compiler.set_executable("compiler_so", nvcc)
                        if isinstance(cflags, dict):
                            cflags = COMMON_NVCC_FLAGS + cflags["nvcc"]
                    elif isinstance(cflags, dict):
                        cflags = cflags["cxx"]
                    cflags += ["-std=c++17"]

                    original_compile(obj, src, ext, cc_args, cflags, pp_opts)
                finally:
                    # Put the original compiler back in place.
                    self.compiler.set_executable("compiler_so", original_compiler)

            self.compiler._compile = unix_wrap_single_compile
            super().build_extensions()

else:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (
                1024**3
            )  # free memory in GB
            max_num_jobs_memory = int(
                free_memory_gb / 9
            )  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


def get_platform():
    """
    Returns the platform name as used in wheel filenames.
    """
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    elif sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    else:
        raise ValueError("Unsupported platform: {}".format(sys.platform))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    # warn instead of error because user could be downloading prebuilt wheels, so nvcc won't be necessary
    # in that case.
    warnings.warn(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def nvcc_threads_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return ["--threads", nvcc_threads]


def generate_cuda_sources():
    DTYPE_16 = (["bf16"] if not DISABLE_BF16 else []) + (
        ["fp16"] if not DISABLE_FP16 else []
    )
    HEAD_DIMENSIONS = (
        []
        + ([32] if not DISABLE_HDIM32 else [])
        + ([64] if not DISABLE_HDIM64 else [])
        + ([128] if not DISABLE_HDIM128 else [])
        + ([256] if not DISABLE_HDIM256 else [])
    )
    RAB = [""] + (["_rab"] if not DISABLE_RAB else [])
    RAB_DRAB = [""] + (
        (["_rab_drab", "_rab"])
        if not DISABLE_DRAB
        else ["_rab"]
        if not DISABLE_RAB
        else []
    )
    MASK = [""]
    FP8_MASK = [""]
    if not DISABLE_LOCAL:
        MASK += ["_local"]
        FP8_MASK += ["_local"]
        MASK += ["_local_deltaq"] if not DISABLE_DELTA_Q else []
    if not DISABLE_CAUSAL:
        CAUSAL_MASK = ["_causal"]
        FP8_MASK += ["_causal"]
        CONTEXT_MASK = [""] + (["_context"] if not DISABLE_CONTEXT else [])
        TARGET_MASK = [""] + (["_target"] if not DISABLE_TARGET else [])
        MASK += [
            f"{c}{x}{t}"
            for c, x, t in itertools.product(CAUSAL_MASK, CONTEXT_MASK, TARGET_MASK)
        ]
        MASK += ["_causal_deltaq"] if not DISABLE_DELTA_Q else []

    dtype_to_str = {
        "bf16": "cutlass::bfloat16_t",
        "fp16": "cutlass::half_t",
    }

    sources_fwd = []
    fwd_file_head = """
// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_cuda_sources() in setup.py

#include "hstu_fwd_launch_template.h"

template void run_hstu_fwd_<90, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_fwd_params& params, cudaStream_t stream);

    """
    for hdim, dtype, rab, mask in itertools.product(
        HEAD_DIMENSIONS, DTYPE_16, RAB, MASK
    ):
        file_name = f"instantiations/hstu_fwd_hdim{hdim}_{dtype}{rab}{mask}_sm90.cu"
        with open(file_name, "w") as f:
            f.write(
                fwd_file_head.format(
                    dtype_to_str[dtype],
                    hdim,
                    "true" if "_rab" in rab else "false",
                    "true" if "local" in mask else "false",
                    "true" if "causal" in mask else "false",
                    "true" if "context" in mask else "false",
                    "true" if "target" in mask else "false",
                    "true" if "deltaq" in mask else "false",
                )
            )
        sources_fwd.append(file_name)
    if not DISABLE_FP8:
        for hdim, rab, mask in itertools.product(HEAD_DIMENSIONS, RAB, FP8_MASK):
            if hdim == 32:
                continue
            file_name = f"instantiations/hstu_fwd_hdim{hdim}_e4m3{rab}{mask}_sm90.cu"
            with open(file_name, "w") as f:
                f.write(
                    fwd_file_head.format(
                        "cutlass::float_e4m3_t",
                        hdim,
                        "true" if "_rab" in rab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "false",  # context
                        "false",  # target
                        "false",
                    )
                )  # deltaq
            sources_fwd.append(file_name)

    sources_bwd = []
    bwd_file_head = """
// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
// Splitting different head dimensions, data types and masks to different files to speed up
// compilation. This file is auto-generated. See generate_cuda_sources() in setup.py

#include "hstu_bwd_launch_template.h"

template void run_hstu_bwd_<90, {}, {}, {}, {}, {}, {}, {}, {}, {}>
                           (Hstu_bwd_params& params, cudaStream_t stream);

    """
    if not DISABLE_BACKWARD:
        for hdim, dtype, rab_drab, mask in itertools.product(
            HEAD_DIMENSIONS, DTYPE_16, RAB_DRAB, MASK
        ):
            file_name = (
                f"instantiations/hstu_bwd_hdim{hdim}_{dtype}{rab_drab}{mask}_sm90.cu"
            )
            with open(file_name, "w") as f:
                f.write(
                    bwd_file_head.format(
                        dtype_to_str[dtype],
                        hdim,
                        "true" if "_rab" in rab_drab else "false",
                        "true" if "drab" in rab_drab else "false",
                        "true" if "local" in mask else "false",
                        "true" if "causal" in mask else "false",
                        "true" if "context" in mask else "false",
                        "true" if "target" in mask else "false",
                        "true" if "deltaq" in mask else "false",
                    )
                )
            sources_bwd.append(file_name)
        # if not DISABLE_FP8:
        #     for hdim, rab_drab, mask in itertools.product(HEAD_DIMENSIONS, RAB_DRAB, FP8_MASK):
        #         if hdim == 32:
        #             continue
        #         file_name = f"instantiations/hstu_bwd_hdim{hdim}_e4m3{rab_drab}{mask}_sm90.cu"
        #         with open(file_name, "w") as f:
        #             f.write(bwd_file_head.format("90",
        #                                         "cutlass::float_e4m3_t",
        #                                         hdim,
        #                                         "true" if "_rab" in rab_drab else "false",
        #                                         "false", # drab
        #                                         "true" if "local" in mask else "false",
        #                                         "true" if "causal" in mask else "false",
        #                                         "false", # context
        #                                         "false", # target
        #                                         "false")) # deltaq
        #         sources_bwd.append(file_name)

    return sources_fwd + sources_bwd


cmdclass = {}
ext_modules = []

# We want this even if SKIP_CUDA_BUILD because when we run python setup.py sdist we want the .hpp
# files included in the source distribution, in case the user compiles from source.
subprocess.run(["git", "submodule", "update", "--init", "../../../third_party/cutlass"])

cmdclass = []
install_requires = []

if not SKIP_CUDA_BUILD:
    if not ONLY_COMPILE_SO:
        print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])

    check_if_cuda_home_none("--hstu")
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.3"):
        raise RuntimeError("HSTU is only supported on CUDA 12.3 and above")

    cc_flag = []
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90a,code=sm_90a")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True
    repo_dir = Path(this_dir).parent
    cutlass_dir = repo_dir / "../../third_party/cutlass"

    feature_args = (
        []
        + (["-DHSTU_DISABLE_BACKWARD"] if DISABLE_BACKWARD else [])
        + (["-DHSTU_DISABLE_LOCAL"] if DISABLE_LOCAL else [])
        + (["-DHSTU_DISABLE_CAUSAL"] if DISABLE_CAUSAL else [])
        + (["-DHSTU_DISABLE_CONTEXT"] if DISABLE_CONTEXT else [])
        + (["-DHSTU_DISABLE_TARGET"] if DISABLE_TARGET else [])
        + (["-DHSTU_DISABLE_DELTA_Q"] if DISABLE_DELTA_Q else [])
        + (["-DHSTU_DISABLE_RAB"] if DISABLE_RAB else [])
        + (["-DHSTU_DISABLE_DRAB"] if DISABLE_DRAB else [])
        + (["-DHSTU_DISABLE_BF16"] if DISABLE_BF16 else [])
        + (["-DHSTU_DISABLE_FP16"] if DISABLE_FP16 else [])
        + (["-DHSTU_DISABLE_FP8"] if DISABLE_FP8 else [])
        + (["-DHSTU_DISABLE_HDIM32"] if DISABLE_HDIM32 else [])
        + (["-DHSTU_DISABLE_HDIM64"] if DISABLE_HDIM64 else [])
        + (["-DHSTU_DISABLE_HDIM128"] if DISABLE_HDIM128 else [])
        + (["-DHSTU_DISABLE_HDIM256"] if DISABLE_HDIM256 else [])
        + (["-DHSTU_DISABLE_SM8x"] if DISABLE_SM8x else [])
    )

    if DISABLE_BF16 and DISABLE_FP16 and DISABLE_FP8:
        raise ValueError(
            "At least one of DISABLE_BF16, DISABLE_FP16, or DISABLE_FP8 must be False"
        )
    if DISABLE_HDIM32 and DISABLE_HDIM64 and DISABLE_HDIM128 and DISABLE_HDIM256:
        raise ValueError(
            "At least one of DISABLE_HDIM32, DISABLE_HDIM64, DISABLE_HDIM128, or DISABLE_HDIM256 must be False"
        )
    if DISABLE_BACKWARD and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without backward")
    if DISABLE_RAB and not DISABLE_DRAB:
        raise ValueError("Cannot support drab without rab")
    if DISABLE_CAUSAL and not DISABLE_TARGET:
        raise ValueError("Cannot support target without causal")
    if DISABLE_CAUSAL and not DISABLE_CONTEXT:
        raise ValueError("Cannot support context without causal")

    torch_cpp_sources = ["hstu_api.cpp"]
    subprocess.run(["rm", "-rf", "instantiations/*"])
    cuda_sources = generate_cuda_sources()

    nvcc_flags = [
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
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        # "--resource-usage",  # printing out number of registers
        # "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-lineinfo",
    ]
    if get_platform() == "win_amd64":
        nvcc_flags.extend(
            [
                "-D_USE_MATH_DEFINES",  # for M_LN2
                "-Xcompiler=/Zc:__cplusplus",  # sets __cplusplus correctly, CUTLASS_CONSTEXPR_IF_CXX17 needed for cutlass::gcd
            ]
        )
    include_dirs = [
        Path(this_dir),
        cutlass_dir / "include",
    ]

    sources = None
    if ONLY_COMPILE_SO:
        sources = cuda_sources
    else:
        sources = torch_cpp_sources + cuda_sources

    ext_modules.append(
        CUDAExtension(
            name="hstu_hopper_cuda",
            sources=sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"] + feature_args,
                "nvcc": nvcc_threads_args() + nvcc_flags + cc_flag + feature_args,
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"],
        )
    )


setup(
    name=PACKAGE_NAME,
    version="0.1.0" + "+cu" + str(bare_metal_version),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    author="NVIDIA-DevTech",
    py_modules=["hstu_attn_interface"],
    description="HSTU Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
)
