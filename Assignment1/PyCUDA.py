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
import numpy as np
from CudaModule import CudaModule
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # List all main methods
    all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu', 'add_gpuarray_no_kernel', 'add_gpuarray_using_kernel']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods[0:6]

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:8]

    # Create an instance of the CudaModule class
    graphicscomputer = CudaModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    for current_operation in valid_operations:
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        arr_avg_total_cpu_multi_thread_time = np.array([])
        arr_avg_total_gpu_global_time = np.array([])
        arr_avg_total_gpu_global_nocopy_time = np.array([])
        arr_avg_total_gpu_shared_time = np.array([])
        arr_avg_total_gpu_shared_nocopy_time = np.array([])
        arr_avg_total_gpu_constant_time = np.array([])
        arr_avg_total_gpu_constant_nocopy_time = np.array([])
        arr_avg_total_gpu_texture_time = np.array([])
        arr_avg_total_gpu_texture_nocopy_time = np.array([])
        arr_avg_total_gpu_register_time = np.array([])
        arr_avg_total_gpu_register_nocopy_time = np.array([])
        
        for vector_size in valid_vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            # [TODO: Students should write Code]
            # Add for the rest of the methods
            arr_total_cpu_multi_thread_time = np.array([])
            arr_total_gpu_global_time = np.array([])
            arr_total_gpu_global_nocopy_time = np.array([])
            arr_total_gpu_shared_time = np.array([])
            arr_total_gpu_shared_nocopy_time = np.array([])
            arr_total_gpu_constant_time = np.array([])
            arr_total_gpu_constant_nocopy_time = np.array([])
            arr_total_gpu_texture_time = np.array([])
            arr_total_gpu_texture_nocopy_time = np.array([])
            arr_total_gpu_register_time = np.array([])
            arr_total_gpu_register_nocopy_time = np.array([])


            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32)
            b = 3 # Choose any number you desire
            b_number_np = np.float32(b) # storing as number having value b with datatype Float32
            b_array_np = b*np.ones(vector_size).astype(np.float32) # storing as array with all elements having equal value b as datatype Float32
            percentdone = 0
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if(current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if(current_method == 'CPU Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if(current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        # [TODO: Students should write Code]
                        # Add for the rest of the methods
                        if (current_method == 'CPU_MultiThread_Add'):
                            c_np_cpu_multi_thread_add, cpu_time_multi_thread = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_np_cpu_multi_thread_add - c_np_cpu_add
                            arr_total_cpu_multi_thread_time = np.append(arr_total_cpu_multi_thread_time, cpu_time_multi_thread)
                        if (current_method == 'GPU_Global_Add'):
                            c_gpu_global_add, gpu_global_time, gpu_global_nocopy_time = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu_global_add - c_np_cpu_add
                            arr_total_gpu_global_time = np.append(arr_total_gpu_global_time, gpu_global_time)
                            arr_total_gpu_global_nocopy_time = np.append(arr_total_gpu_global_nocopy_time, gpu_global_nocopy_time)
                        if (current_method == 'GPU_Shared_Add'):
                            c_gpu_shared_add, gpu_shared_time, gpu_shared_nocopy_time = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu_shared_add - c_np_cpu_add
                            arr_total_gpu_shared_time = np.append(arr_total_gpu_shared_time, gpu_shared_time)
                            arr_total_gpu_shared_nocopy_time = np.append(arr_total_gpu_shared_nocopy_time, gpu_shared_nocopy_time)
                        if (current_method == 'GPU_Constant_Add'):
                            c_gpu_constant_add, gpu_constant_time, gpu_constant_nocopy_time = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu_constant_add - c_np_cpu_add
                            arr_total_gpu_constant_time = np.append(arr_total_gpu_constant_time, gpu_constant_time)
                            arr_total_gpu_constant_nocopy_time = np.append(arr_total_gpu_constant_nocopy_time, gpu_constant_nocopy_time)
                        if (current_method == 'GPU_Texture_Add'):
                            c_gpu_texture_add, gpu_texture_time, gpu_texture_nocopy_time = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu_texture_add - c_np_cpu_add
                            arr_total_gpu_texture_time = np.append(arr_total_gpu_texture_time, gpu_texture_time)
                            arr_total_gpu_texture_nocopy_time = np.append(arr_total_gpu_texture_nocopy_time, gpu_texture_nocopy_time)
                        if (current_method == 'GPU_Register_Add'):
                            c_gpu_register_add, gpu_register_time, gpu_register_nocopy_time = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu_register_add - c_np_cpu_add
                            arr_total_gpu_register_time = np.append(arr_total_gpu_register_time, gpu_register_time)
                            arr_total_gpu_register_nocopy_time = np.append(arr_total_gpu_register_nocopy_time, gpu_register_nocopy_time)
                        #else:
                            #print("Unknown method:", current_method)
                        

                       
                        total_diff = sum_diff.sum()
                        if (total_diff != 0):
                            print (current_method + " " + current_operation + "sum mismatch")
                            print (total_diff)
                            
            avg_total_cpu_time = ((arr_total_cpu_time.sum())/50)
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = ((arr_total_cpu_loop_time.sum())/50)
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)

            # [TODO: Students should write Code]
            # Add for the rest of the methods
            avg_total_cpu_multi_thread_time = (arr_total_cpu_multi_thread_time.sum())/50
            arr_avg_total_cpu_multi_thread_time = np.append(arr_avg_total_cpu_multi_thread_time, avg_total_cpu_multi_thread_time)
            avg_total_gpu_global_time = (arr_total_gpu_global_time.sum())/50
            arr_avg_total_gpu_global_time = np.append(arr_avg_total_gpu_global_time, avg_total_gpu_global_time)
            avg_total_gpu_global_nocopy_time = (arr_total_gpu_global_nocopy_time.sum())/50
            arr_avg_total_gpu_global_nocopy_time = np.append(arr_avg_total_gpu_global_nocopy_time, avg_total_gpu_global_nocopy_time)
            avg_total_gpu_shared_time = (arr_total_gpu_shared_time.sum())/50
            arr_avg_total_gpu_shared_time = np.append(arr_avg_total_gpu_shared_time, avg_total_gpu_shared_time)
            avg_total_gpu_shared_nocopy_time = (arr_total_gpu_shared_nocopy_time.sum())/50
            arr_avg_total_gpu_shared_nocopy_time = np.append(arr_avg_total_gpu_shared_nocopy_time, avg_total_gpu_shared_nocopy_time)
            avg_total_gpu_constant_time = (arr_total_gpu_constant_time.sum())/50
            arr_avg_total_gpu_constant_time = np.append(arr_avg_total_gpu_constant_time, avg_total_gpu_constant_time)
            avg_total_gpu_constant_nocopy_time = (arr_total_gpu_constant_nocopy_time.sum())/50
            arr_avg_total_gpu_constant_nocopy_time = np.append(arr_avg_total_gpu_constant_nocopy_time, avg_total_gpu_constant_nocopy_time)
            avg_total_gpu_texture_time = (arr_total_gpu_texture_time.sum())/50
            arr_avg_total_gpu_texture_time = np.append(arr_avg_total_gpu_texture_time, avg_total_gpu_texture_time)
            avg_total_gpu_texture_nocopy_time = (arr_total_gpu_texture_nocopy_time.sum())/50
            arr_avg_total_gpu_texture_nocopy_time = np.append(arr_avg_total_gpu_texture_nocopy_time, avg_total_gpu_texture_nocopy_time)
            avg_total_gpu_register_time = (arr_total_gpu_register_time.sum())/50
            arr_avg_total_gpu_register_time = np.append(arr_avg_total_gpu_register_time, avg_total_gpu_register_time)
            avg_total_gpu_register_nocopy_time = (arr_total_gpu_register_nocopy_time.sum())/50
            arr_avg_total_gpu_register_nocopy_time = np.append(arr_avg_total_gpu_register_nocopy_time, avg_total_gpu_register_nocopy_time)

        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        print(current_operation + " The CPU Multi-thread times are")
        print(arr_avg_total_cpu_multi_thread_time)
        print(current_operation + " The GPU Global times are")
        print(arr_avg_total_gpu_global_time)
        print(current_operation + " The GPU Global no copy times are")
        print(arr_avg_total_gpu_global_nocopy_time)

        # Code for Plotting the results 
        vector_sizes = [10**i for i in range(1, 9)]
        plt.figure()
        plt.plot(vector_sizes, arr_avg_total_cpu_time, label='CPU Add')
        plt.plot(vector_sizes, arr_avg_total_cpu_multi_thread_time, label='CPU Multi-threaded Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_global_time, label='GPU Global Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_shared_time, label='GPU Shared Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_constant_time, label='GPU Constant Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_texture_time, label='GPU Texture Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_register_time, label='GPU Register Add')
        plt.xlabel('Vector Size (10^n)')
        plt.ylabel('Average Execution Time (including memory transfer)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(vector_sizes, [f'$10^{i}$' for i in range(1, 9)])
        plt.legend()
        plt.title('Average Execution Time vs Vector Size (Including Memory Transfer)')
        plt.show()

        plt.figure()
        plt.plot(vector_sizes, arr_avg_total_gpu_global_nocopy_time, label='GPU Global Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_shared_nocopy_time, label='GPU Shared Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_constant_nocopy_time, label='GPU Constant Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_texture_nocopy_time, label='GPU Texture Add')
        plt.plot(vector_sizes, arr_avg_total_gpu_register_nocopy_time, label='GPU Register Add')
        plt.xlabel('Vector Size (10^n)')
        plt.ylabel('Average Execution Time (excluding memory transfer)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(vector_sizes, [f'$10^{i}$' for i in range(1, 9)])
        plt.legend()
        plt.title('Average Execution Time vs Vector Size (Excluding Memory Transfer)')
        plt.show()
