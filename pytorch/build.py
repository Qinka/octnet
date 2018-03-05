import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(__file__)

headers = ['octnet/src/types.h','octnet/src/func_cpu.h']
with_cuda = False

if torch.cuda.is_available():
    print("with cuda")
    headers += ['octnet/src/func_gpu.h']
    with_cuda = True

oc_p = os.environ['OCTNET_PATH']

ffi = create_extension(
    'octnet._ext.octnet',
    package= True,
    headers = headers,
    sources = [],
    define_macros = [],
    relative_to= __file__,
    libraries = ['octnet_core'],
    library_dirs = [oc_p+'/lib',oc_p+'/bin',oc_p],
    with_cuda= with_cuda,
)

if __name__ == '__main__':
    ffi.build()