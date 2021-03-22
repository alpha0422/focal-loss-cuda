#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import subprocess
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Refer to https://github.com/NVIDIA/apex/blob/e2083df5eb96643c61613b9df48dd4eea6b07690/setup.py


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor


if not torch.cuda.is_available():
    print('\nWarning: Torch did not find available GPUs on this system.\n',
          'If your intention is to cross-compile, this is not an error.\n'
          'By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n'
          'Volta (compute capability 7.0), Turing (compute capability 7.5),\n'
          'and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n'
          'If you wish to cross-compile for a single specific architecture,\n'
          'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n')
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        _, bare_metal_major, _ = get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME)
        if int(bare_metal_major) == 11:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

print("\n\ntorch.__version__  = {}".format(torch.__version__))
print("Build with CUDA Arch: {}\n\n".format(
    os.environ["TORCH_CUDA_ARCH_LIST"]))

setup(
    name='focal_loss',
    version='0.1',
    description='High performance PyTorch CUDA focal loss.',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('focal_loss_cuda', [
            'csrc/focal_loss_cuda.cpp',
            'csrc/focal_loss_cuda_kernel.cu',
        ],
            extra_compile_args={
            'cxx': ['-O3', ],
            'nvcc':['-O3', '-lineinfo', '-res-usage', '--use_fast_math']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    test_suite="tests",
)
