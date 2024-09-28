#!/usr/bin/env python

"""
.
.
.
Python Code
.
.
.
"""

# The code in this file is part of the instructor-provided template for Assignment-1, Fall 2024.

# Import relevant libraries
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time
import pycuda.gpuarray as gpuarray

class CudaModule:
    def __init__(self):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the functions
        # you will call from this class.
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compiles Kernel in Source Module to be used by functions across the class.
        """
        # Define your kernel below.
        kernelwrapper = """
        __global__ void Add_two_vectors_GPU(float *a, float *b, float *c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }

        __global__ void Add_to_each_element_GPU(float *a, float b, float *c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b;
            }
        }
        """
        return SourceModule(kernelwrapper)

    def add_device_mem_gpu(self, a, b, length, is_b_a_vector):
        # Ensure length is a standard int
        length = int(length)

        # Event objects to mark the start and end points
        start_total = time.time()
        start_gpu = cuda.Event()
        end_gpu = cuda.Event()

        # Device memory allocation for input and output arrays
        a_gpu = cuda.mem_alloc(a.nbytes)
        c_gpu = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_gpu, a)

        if is_b_a_vector:
            b_gpu = cuda.mem_alloc(b.nbytes)
            cuda.memcpy_htod(b_gpu, b)
        else:
            b_float = np.float32(b)

        # Get grid and block dim
        block_size = 256  # Standard int
        grid_size = int((length + block_size - 1) // block_size)  # Convert to int

        # Call the kernel function from the compiled module
        if is_b_a_vector:
            func = self.mod.get_function("Add_two_vectors_GPU")
            start_gpu.record()
            func(a_gpu, b_gpu, c_gpu, np.int32(length), block=(block_size, 1, 1), grid=(grid_size, 1))
            end_gpu.record()
        else:
            func = self.mod.get_function("Add_to_each_element_GPU")
            start_gpu.record()
            func(a_gpu, b_float, c_gpu, np.int32(length), block=(block_size, 1, 1), grid=(grid_size, 1))
            end_gpu.record()

        # Wait for the event to complete
        end_gpu.synchronize()
        gpu_time = start_gpu.time_till(end_gpu) * 1e-3  # Convert to seconds

        # Copy result from device to the host
        c = np.empty_like(a)
        cuda.memcpy_dtoh(c, c_gpu)
        total_time = time.time() - start_total

        # Return a tuple of output of addition and time taken to execute the operation.
        return c, total_time, gpu_time

    def add_host_mem_gpu(self, a, b, length, is_b_a_vector):
        length = int(length)
        # Event objects to mark the start and end points
        start_total = time.time()

        # Get grid and block dim
        block_size = 256
        grid_size = int((length + block_size - 1) // block_size)

        # Call the kernel function from the compiled module
        c = np.empty_like(a)
        if is_b_a_vector:
            func = self.mod.get_function("Add_two_vectors_GPU")
            func(cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(length),
                block=(block_size, 1, 1), grid=(grid_size, 1))
        else:
            func = self.mod.get_function("Add_to_each_element_GPU")
            b_float = np.float32(b)
            func(cuda.In(a), b_float, cuda.Out(c), np.int32(length),
                block=(block_size, 1, 1), grid=(grid_size, 1))

        total_time = time.time() - start_total

        # Return a tuple of output of addition and time taken to execute the operation.
        return c, total_time


    def add_gpuarray_no_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel.
        The operation is defined using numpy-like syntax.
        Arguments:
            a             : numpy array of size: length
            b             : numpy array of size: length or scalar
            length        : length of numpy arrays
            is_b_a_vector : Boolean indicating if b is a vector
        Returns:
            c             : addition result
            time_incl     : execution time including memory transfer
            time_excl     : execution time excluding memory transfer (Only Operation)
        """
        # Event objects to mark start and end points
        start_total = time.time()
        start_gpu = cuda.Event()
        end_gpu = cuda.Event()

        # Allocate device memory using gpuarray class
        a_gpu = gpuarray.to_gpu(a)
        if is_b_a_vector:
            b_gpu = gpuarray.to_gpu(b)
        else:
            b_float = np.float32(b)

        # Record execution time and execute operation with numpy syntax
        start_gpu.record()
        if is_b_a_vector:
            c_gpu = a_gpu + b_gpu
        else:
            c_gpu = a_gpu + b_float
        end_gpu.record()

        # Wait for the event to complete
        end_gpu.synchronize()
        gpu_time = start_gpu.time_till(end_gpu) * 1e-3  # Convert to seconds

        # Fetch result from device to host
        c = c_gpu.get()
        total_time = time.time() - start_total

        # Return a tuple of output of addition and time taken to execute the operation.
        return c, total_time, gpu_time

    def add_gpuarray_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that
        you call the kernel function.
        Arguments:
            a             : numpy array of size: length
            b             : numpy array of size: length or scalar
            length        : length of numpy arrays
            is_b_a_vector : Boolean indicating if b is a vector
        Returns:
            c             : addition result
            time_incl     : execution time including memory transfer
            time_excl     : execution time excluding memory transfer (Only Kernel)
        """
        # Create cuda events to mark the start and end of array.
        start_total = time.time()
        start_gpu = cuda.Event()
        end_gpu = cuda.Event()

        # Get function defined in class definition
        if is_b_a_vector:
            b_gpu = gpuarray.to_gpu(b)
        else:
            b_float = np.float32(b)

        # Allocate device memory for a, b, output of addition using gpuarray class
        a_gpu = gpuarray.to_gpu(a)
        c_gpu = gpuarray.empty_like(a_gpu)

        # Get grid and block dim
        block_size = 256
        grid_size = (length + block_size - 1) // block_size

        # Record execution time and execute operation
        if is_b_a_vector:
            func = self.mod.get_function("Add_two_vectors_GPU")
            start_gpu.record()
            func(a_gpu.gpudata, b_gpu.gpudata, c_gpu.gpudata, np.int32(length),
                 block=(block_size, 1, 1), grid=(grid_size, 1))
            end_gpu.record()
        else:
            func = self.mod.get_function("Add_to_each_element_GPU")
            start_gpu.record()
            func(a_gpu.gpudata, b_float, c_gpu.gpudata, np.int32(length),
                 block=(block_size, 1, 1), grid=(grid_size, 1))
            end_gpu.record()

        # Wait for the event to complete
        end_gpu.synchronize()
        gpu_time = start_gpu.time_till(end_gpu) * 1e-3  # Convert to seconds

        # Fetch result from device to host
        c = c_gpu.get()
        total_time = time.time() - start_total

        # Return a tuple of output of addition and time taken to execute the operation.
        return c, total_time, gpu_time

    def CPU_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             : 1st Vector
            b             : number or vector of equal numbers with same length as a
            length        : length of vector a
            is_b_a_vector : Boolean Describing if b is a vector or a number
        """
        start = time.time()
        c = a + b
        end = time.time()

        return c, end - start

    def CPU_Loop_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             : 1st Vector
            b             : number or vector of equal numbers with same length as a
            length        : length of vector a
            is_b_a_vector : Boolean Describing if b is a vector or a number
        """

        start = time.time()
        c = np.empty_like(a)
        for index in np.arange(0, length):
            if (is_b_a_vector == True):
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        return c, end - start

