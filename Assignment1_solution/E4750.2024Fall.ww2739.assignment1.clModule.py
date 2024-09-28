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

"""
The code in this file is part of the instructor-provided template for Assignment-1, Fall 2024.
"""

# import relevant libraries
import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import time

class clModule:
    def __init__(self):
        """
        **Do not modify this code**
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code.
        """

        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()

        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()

        # Create Context:
        self.ctx = cl.Context(devs)

        # Setup Command Queue:
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # kernel - will not be provided for future assignments!
        kernel_code = """

            __kernel void Add_two_vectors_GPU(__global float* c, __global float* a, __global float* b, const unsigned int n){

                unsigned int i = get_global_id(0);
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }

            __kernel void Add_to_each_element_GPU(__global float* c, __global float* a, float b, const unsigned int n){

                unsigned int i = get_global_id(0);
                if (i < n){
                    c[i] = a[i] + b;
                }
            }
        """

        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()

    def deviceAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.array class
        Arguments:
            a             :   1st Vector
            b             :   number or vector of equal numbers with same length as a
            length        :   length of vector a
            is_b_a_vector :   Boolean Describing if b is a vector or a number
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
        # device memory allocation
        t_start = time.time()
        a_device = cl_array.to_device(self.queue, a)
        c_device = cl_array.empty_like(a_device)
        # execute operation.
        if is_b_a_vector:
            # Use `Add_two_vectors_GPU` Kernel.
            b_device = cl_array.to_device(self.queue, b)
            event = self.prg.Add_two_vectors_GPU(self.queue, (length,), None, c_device.data, a_device.data, b_device.data, np.uint32(length))
        else:
            # Use `Add_to_each_element_GPU` Kernel
            event = self.prg.Add_to_each_element_GPU(self.queue, (length,), None, c_device.data, a_device.data, np.float32(b), np.uint32(length))

        # wait for execution to complete.
        event.wait()
        # Copy output from GPU to CPU [Use .get() method]
        c = c_device.get()
        # Record execution time.
        time_ = time.time() - t_start
        # return a tuple of output of addition and time taken to execute the operation.
        return (c, time_)

    def bufferAdd(self, a, b, length, is_b_a_vector):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # Create three buffers (plans for areas of memory on the device)
        m_flag = cl.mem_flags
        start = time.time()
        a_buffer = cl.Buffer(self.ctx, m_flag.READ_ONLY | m_flag.COPY_HOST_PTR, hostbuf=a)
        c_buffer = cl.Buffer(self.ctx, m_flag.WRITE_ONLY, a.nbytes)

        if is_b_a_vector:
            # Use `Add_two_vectors_GPU` Kernel.
            b_buffer = cl.Buffer(self.ctx, m_flag.READ_ONLY | m_flag.COPY_HOST_PTR, hostbuf=b)
            event = self.prg.Add_two_vectors_GPU(self.queue, (length,), None, c_buffer, a_buffer, b_buffer, np.uint32(length))
        else:
            # Use `Add_to_each_element_GPU` Kernel
            event = self.prg.Add_to_each_element_GPU(self.queue, (length,), None, c_buffer, a_buffer, np.float32(b), np.uint32(length))

        # Wait for execution to complete.
        event.wait()
        # Copy output from GPU to CPU [Use enqueue_copy]
        c = np.empty_like(a)
        cl.enqueue_copy(self.queue, c, c_buffer)
        # Record execution time.
        end = time.time()
        # return a tuple of output of addition and time taken to execute the operation.
        return c, (end - start)

    def CPU_numpy_Add(self, a, b, length, is_b_a_vector):
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
        for index in np.arange(0, length):
            if is_b_a_vector:
                c[index] = a[index] + b[index]
            else:
                c[index] = a[index] + b
        end = time.time()

        return c, end - start
