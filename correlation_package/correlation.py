import os
import sys
from torch.utils.cpp_extension import load

# 1. Get the absolute path to the 'correlation_package' directory
# This ensures it works flawlessly inside Jupyter/Colab regardless of your current working directory
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

print("corrrelation_package compile CUDA", current_dir)

# If correlation.py is inside a subfolder, we make sure we target the correct paths
# (Adjust the joins below if your folders have different exact names)

cuda_source_path = os.path.join(current_dir, '..', 'correlation_package', 'correlation_cuda.cc') 

# Note: If they are simply sibling files/folders, make sure the path points exactly to the .cc file.

# 2. Compile and load the CUDA extension at runtime
# This will take a few moments on the first execution, then use a cached build on subsequent runs.
try:
    compiled_module = load(
        name="correlation_cuda",
        sources=[cuda_source_path, os.path.join(current_dir, '..', 'correlation_package', 'correlation_cuda_kernel.cu')],
        verbose=True
    )
    print("CUDA Extension compiled and loaded successfully!")
except Exception as e:
    print(f"Failed to compile CUDA extension. Error: {e}")
    print("Ensure a CUDA-capable GPU runtime is active in your notebook.")
    
sys.modules['correlation_cuda'] = compiled_module

# 3. You can now use the compiled functions directly from the object
# Example: output = correlation_cuda.forward(input1, input2)

import torch
from torch.nn.modules.module import Module
from torch.autograd import Function
import correlation_cuda
import os

class CorrelationFunction(Function):

    # def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
    #     super(CorrelationFunction, self).__init__()
    #     self.pad_size = pad_size
    #     self.kernel_size = kernel_size
    #     self.max_displacement = max_displacement
    #     self.stride1 = stride1
    #     self.stride2 = stride2
    #     self.corr_multiply = corr_multiply
    #     # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement,stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                     pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      ctx.pad_size, ctx.kernel_size, ctx.max_displacement, ctx.stride1, ctx.stride2, ctx.corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    # @staticmethod
    def forward(self, input1, input2):

        input1 = input1.contiguous()
        input2 = input2.contiguous()
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return result

