import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

sources = ["src/deform_conv_2d.c"]
headers = ["src/deform_conv_2d.h"]
defines = []
with_cuda = False

if torch.cuda.is_available():
    print("Including CUDA code.")
    sources += ["src/deform_conv_2d_cuda.c"]
    headers += ["src/deform_conv_2d_cuda.h"]
    defines += [("WITH_CUDA", None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
extra_objects = ["src/deform_conv_2d_cuda_kernel.cu.so"]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    "_ext.deform_conv_2d",
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=["-std=c++11"],
)

assert torch.cuda.is_available(), "Please install CUDA for GPU support."
ffi.build()
