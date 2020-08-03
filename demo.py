import numpy as np
import time

from numba import cfunc, types, carray, objmode

c_sig = types.void(types.CPointer(types.uint8),
                   types.CPointer(types.uint8),
                   types.int64)

@cfunc(c_sig, nopython=True)
def hello_cfunc(in_ptr, out_ptr, size):
    in_arr = carray(in_ptr, size)
    out_arr = carray(out_ptr, size)

    out_arr[:] = 255

#print(hello_cfunc.address)
#print(hello_cfunc.inspect_llvm())

import nvidia.dali.plugin_manager as plugin_manager
plugin_manager.load_library('/home/agartner/dali-numba-plugin/libcustomdummy.so')
image_dir = "/home/agartner/DALI/docs/examples/data/images"
batch_size = 8

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        self.decode = ops.ImageDecoder(device='cpu', output_type=types.RGB)
        self.custom = ops.CustomDummy(fn_ptr=hello_cfunc.address)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        images = self.custom(images)
        return (images, labels)

pipe = SimplePipeline(batch_size, 1, 0)
pipe.build()

pipe_out = pipe.run()
images, labels = pipe_out

image = np.array(images[0])
print(f'min: {image.min()}')
print(f'max: {image.max()}')
print(f'std: {image.std()}')