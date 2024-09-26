"""
The code in this file is part of the instructor-provided template for Assignment-1, Fall 2024. 
"""

# import relevant libraries
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import numpy as np
import time


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
        # define your kernel below.
        kernelwrapper ="""
        __global__ void Add_two_vectors_GPU(float* c, float* a, float* b, const unsigned int n)
        {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b[i];
            }
        }

        __global__ void Add_to_each_element_GPU(float* c, float* a, float b, const unsigned int n){
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                c[i] = a[i] + b;
            }
        }
        """
        return SourceModule(kernelwrapper)

    
    def add_device_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        
        # Event objects to mark the start and end points
        event_start = cuda.Event()
        event_end = cuda.Event()
        # Device memory allocation for input and output arrays
        d_a = cuda.mem_alloc(a.nbytes)
        d_b = cuda.mem_alloc(b.nbytes)
        d_c = cuda.mem_alloc(a.nbytes)
        
        t_start = time.time()        
        # Copy data from host to device
        cuda.memcpy_htod(d_a, a)
        cuda.memcpy_htod(d_b, b)
        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            func = self.mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            func = self.mod.get_function("Add_to_each_element_GPU")

        # Get grid and block dim
        block_size= 256
        grid_size = (length + block_size - 1) // block_size
        block = (block_size, 1, 1)
        grid = (grid_size, 1)
        # Record execution time and call the kernel loaded to the device
        event_start.record()
        if (is_b_a_vector == True):
            func(d_c, d_a, d_b, np.uint32(length), block=block, grid=grid)
        else:
            func(d_c, d_a, np.float32(b), np.uint32(length), block=block, grid=grid)
        # Wait for the event to complete
        event_end.record()
        event_end.synchronize()
        # Copy result from device to the host
        c = np.empty_like(a)
        cuda.memcpy_dtoh(c, d_c)
        # Record the end time and calculate the time taken
        t_end = time.time()
        time = t_end - t_start
        # return a tuple of output of addition and time taken to execute the operation.
        return c, time
        #pass

    
    def add_host_mem_gpu(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method for both cases of is_b_a_vector]
        # Event objects to mark the start and end points
        event_start = cuda.Event()
        event_end = cuda.Event()
        # Get grid and block dim
        block_size= 256
        grid_size = (length + block_size - 1) // block_size
        block = (block_size, 1, 1)
        grid = (grid_size, 1)
        a = a.astype(np.float32)
        c = np.empty_like(a)
        # Call the kernel function from the compiled module
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            k_func = self.mod.get_function("Add_two_vectors_GPU")
            event_start.record()
            t_start = time.time()
            k_func(cuda.Out(c), cuda.In(a), cuda.In(b), np.uint32(length), block=block, grid=grid)
            t_end = time.time()
        else:
            # Use `Add_to_each_element_GPU` Kernel
            k_func = self.mod.get_function("Add_to_each_element_GPU")
            event_start.record()
            t_start = time.time()
            k_func(cuda.Out(c), cuda.In(a), np.float32(b), np.uint32(length), block=block, grid=grid)
            t_end = time.time()
   
        # Record execution time and call the kernel loaded to the device
        time = t_end - t_start

        # Wait for the event to complete
        event_end.record()
        event_end.synchronize()
        
        # return a tuple of output of addition and time taken to execute the operation.
        return c, time
        #pass


    def add_gpuarray_no_kernel(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead) and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]
        # Event objects to mark start and end points
        event_start = cuda.Event()
        event_end = cuda.Event()
        # Allocate device memory using gpuarray class        
        a_gpu = gpuarray.to_gpu(a.astype(np.float32))
        # Record execution time and execute operation with numpy syntax
        t_start = time.time()
        
        if (is_b_a_vector == True):
            b_gpu = gpuarray.to_gpu(b.astype(np.float32))
            event_start.record()
            c_gpu = a_gpu + b_gpu
            event_end.record()
        else:
            b_scalar = np.float32(b)
            event_start.record()
            c_gpu = a_gpu + b_scalar
            event_end.record()
            
        # Wait for the event to complete
        event_end.synchronize()
        t_end = time.time()
        # Fetch result from device to host
        c = c_gpu.get()
        # Record the end time and calculate the time taken
        time = t_end - t_start
        # return a tuple of output of addition and time taken to execute the operation.
        return c, time
        #pass
        
    def add_gpuarray(self, a, b, length, is_b_a_vector):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables (use gpuarray.to_gpu instead). In this scenario make sure that 
        you call the kernel function.
        Arguments:
            a                               :   numpy array of size: length
            b                               :   numpy array of size: length
            length                          :   length of numpy arrays 
        Returns
            c                               :   addition result
            time                            :   execution time
        """
        # [TODO: Students should write code for the entire method. Sufficient to be able to do for is_b_a_vector == True case alone. Bonus points if is_b_a_vector == False case is solved by passing a single number to GPUarray and performing the addition]

        # Create cuda events to mark the start and end of array.
        event_start = cuda.Event()
        event_end = cuda.Event()
        # Get function defined in class defination
        if (is_b_a_vector == True):
            # Use `Add_two_vectors_GPU` Kernel.
            k_func = self.mod.get_function("Add_two_vectors_GPU")
        else:
            # Use `Add_to_each_element_GPU` Kernel
            k_func = self.mod.get_function("Add_to_each_element_GPU")
        # Allocate device memory for a, b, output of addition using gpuarray class        
        
        a_gpu = gpuarray.to_gpu(a.astype(np.float32))
        if (is_b_a_vector == True):
            b_gpu = gpuarray.to_gpu(b.astype(np.float32))
        else:
            b_gpu = np.float32(b)
        c_gpu = gpuarray.empty_like(a_gpu)
        # Get grid and block dim
        block_size= 256
        grid_size = (length + block_size - 1) // block_size
        # Record execution time and execute operation
        t_start = time.time()
        event_start.record()
        if (is_b_a_vector == True):
            k_func(c_gpu.gpudata, a_gpu.gpudata, b_gpu.gpudata, np.uint32(length), block=(block_size, 1, 1), grid=(grid_size, 1))
        else:
            k_func(c_gpu.gpudata, a_gpu.gpudata, b_gpu, np.uint32(length), block=(block_size, 1, 1), grid=(grid_size, 1))
        # Wait for the event to complete
        event_end.record()
        event_end.synchronize()
        # Fetch result from device to host
        c = c_gpu.get()

        # Record the end time and calculate the time taken
        t_end = time.time()
        time = t_end - t_start
        # return a tuple of output of addition and time taken to execute the operation.
        return c, time
        #pass

    def CPU_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """
        start = time.time()
        c = a + b
        end = time.time()

        return c, end - start

    def CPU_Loop_Add(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition on host(CPU) using numpy add method
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        """

        start = time.time()
        c = np.empty_like(a)
        for index in np.arange(0,length):
            if (is_b_a_vector == True):
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        
        return c, end - start

