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
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
module_name = "E4750.2024Fall.ww2739.assignment1.CudaModule"
module_path = "/home/ww2739/E4750.2024Fall.ww2739.assignment1.CudaModule.py"

spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

if __name__ == "__main__":

    # List all main methods
    all_main_methods = ['CPU Add', 'CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu',
                        'add_gpuarray_kernel', 'add_gpuarray_no_kernel']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10 ** np.arange(1, 9)
    # List iteration indexes
    iteration_indexes = np.arange(1, 50)

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid methods to perform (populate as you complete the methods).
    valid_main_methods = all_main_methods  # Now includes all methods

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:6]  # Up to 10^6

    # Create an instance of the CudaModule class
    graphicscomputer = module.CudaModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    # Start the nested loop iteration
    for current_operation in valid_operations:
        # Initialize arrays to store the average times for each method
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        arr_avg_total_add_device_mem_gpu_time_incl = np.array([])
        arr_avg_total_add_device_mem_gpu_time_excl = np.array([])
        arr_avg_total_add_host_mem_gpu_time = np.array([])
        arr_avg_total_add_gpuarray_kernel_time_incl = np.array([])
        arr_avg_total_add_gpuarray_kernel_time_excl = np.array([])
        arr_avg_total_add_gpuarray_no_kernel_time_incl = np.array([])
        arr_avg_total_add_gpuarray_no_kernel_time_excl = np.array([])
        # Loop over the valid vector sizes
        for vector_size in valid_vector_sizes:
            # Convert the vector_size to an integer
            vector_size=int(vector_size)
            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            arr_total_add_device_mem_gpu_time_incl = np.array([])
            arr_total_add_device_mem_gpu_time_excl = np.array([])
            arr_total_add_host_mem_gpu_time = np.array([])
            arr_total_add_gpuarray_kernel_time_incl = np.array([])
            arr_total_add_gpuarray_kernel_time_excl = np.array([])
            arr_total_add_gpuarray_no_kernel_time_incl = np.array([])
            arr_total_add_gpuarray_no_kernel_time_excl = np.array([])

            print("vectorlength")
            print(vector_size)

            a_array_np = np.arange(1, vector_size + 1).astype(np.float32)
            b = 3  # Choose any number you desire
            b_number_np = np.float32(b)  # storing as number having value b with datatype Float32
            b_array_np = b * np.ones(vector_size).astype(np.float32)  # storing as array with all elements having equal value b as datatype Float32

            # Loop over the iterations
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if (current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if (current_method == 'CPU Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if (current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        elif (current_method == 'add_device_mem_gpu'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_device_mem_gpu(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_device_mem_gpu_time_incl = np.append(arr_total_add_device_mem_gpu_time_incl, total_time)
                            arr_total_add_device_mem_gpu_time_excl = np.append(arr_total_add_device_mem_gpu_time_excl, gpu_time)
                        elif (current_method == 'add_host_mem_gpu'):
                            c_gpu, total_time = graphicscomputer.add_host_mem_gpu(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_host_mem_gpu_time = np.append(arr_total_add_host_mem_gpu_time, total_time)
                        elif (current_method == 'add_gpuarray_kernel'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_gpuarray_kernel(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_gpuarray_kernel_time_incl = np.append(arr_total_add_gpuarray_kernel_time_incl, total_time)
                            arr_total_add_gpuarray_kernel_time_excl = np.append(arr_total_add_gpuarray_kernel_time_excl, gpu_time)
                        elif (current_method == 'add_gpuarray_no_kernel'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_gpuarray_no_kernel(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_gpuarray_no_kernel_time_incl = np.append(arr_total_add_gpuarray_no_kernel_time_incl, total_time)
                            arr_total_add_gpuarray_no_kernel_time_excl = np.append(arr_total_add_gpuarray_no_kernel_time_excl, gpu_time)
                        else:
                            print("Unknown method:", current_method)
                        total_diff = np.sum(np.abs(sum_diff))
                        if (total_diff != 0):
                            print(current_method + " " + current_operation + " sum mismatch")
                            print(total_diff)

            # Calculate the average times for each method
            avg_total_cpu_time = arr_total_cpu_time.mean()
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = arr_total_cpu_loop_time.mean()
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            avg_total_add_device_mem_gpu_time_incl = arr_total_add_device_mem_gpu_time_incl.mean()
            arr_avg_total_add_device_mem_gpu_time_incl = np.append(arr_avg_total_add_device_mem_gpu_time_incl, avg_total_add_device_mem_gpu_time_incl)
            avg_total_add_device_mem_gpu_time_excl = arr_total_add_device_mem_gpu_time_excl.mean()
            arr_avg_total_add_device_mem_gpu_time_excl = np.append(arr_avg_total_add_device_mem_gpu_time_excl, avg_total_add_device_mem_gpu_time_excl)
            avg_total_add_host_mem_gpu_time = arr_total_add_host_mem_gpu_time.mean()
            arr_avg_total_add_host_mem_gpu_time = np.append(arr_avg_total_add_host_mem_gpu_time, avg_total_add_host_mem_gpu_time)
            avg_total_add_gpuarray_kernel_time_incl = arr_total_add_gpuarray_kernel_time_incl.mean()
            arr_avg_total_add_gpuarray_kernel_time_incl = np.append(arr_avg_total_add_gpuarray_kernel_time_incl, avg_total_add_gpuarray_kernel_time_incl)
            avg_total_add_gpuarray_kernel_time_excl = arr_total_add_gpuarray_kernel_time_excl.mean()
            arr_avg_total_add_gpuarray_kernel_time_excl = np.append(arr_avg_total_add_gpuarray_kernel_time_excl, avg_total_add_gpuarray_kernel_time_excl)
            avg_total_add_gpuarray_no_kernel_time_incl = arr_total_add_gpuarray_no_kernel_time_incl.mean()
            arr_avg_total_add_gpuarray_no_kernel_time_incl = np.append(arr_avg_total_add_gpuarray_no_kernel_time_incl, avg_total_add_gpuarray_no_kernel_time_incl)
            avg_total_add_gpuarray_no_kernel_time_excl = arr_total_add_gpuarray_no_kernel_time_excl.mean()
            arr_avg_total_add_gpuarray_no_kernel_time_excl = np.append(arr_avg_total_add_gpuarray_no_kernel_time_excl, avg_total_add_gpuarray_no_kernel_time_excl)
        # Print the results
        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        print(current_operation + " The add_device_mem_gpu times including memory transfer are")
        print(arr_avg_total_add_device_mem_gpu_time_incl)
        print(current_operation + " The add_device_mem_gpu times excluding memory transfer are")
        print(arr_avg_total_add_device_mem_gpu_time_excl)
        print(current_operation + " The add_host_mem_gpu times are")
        print(arr_avg_total_add_host_mem_gpu_time)
        print(current_operation + " The add_gpuarray_kernel times including memory transfer are")
        print(arr_avg_total_add_gpuarray_kernel_time_incl)
        print(current_operation + " The add_gpuarray_kernel times excluding memory transfer are")
        print(arr_avg_total_add_gpuarray_kernel_time_excl)
        print(current_operation + " The add_gpuarray_no_kernel times including memory transfer are")
        print(arr_avg_total_add_gpuarray_no_kernel_time_incl)
        print(current_operation + " The add_gpuarray_no_kernel times excluding memory transfer are")
        print(arr_avg_total_add_gpuarray_no_kernel_time_excl)

    # Now exclude the slower of the two CPU cases and extend the analysis to vector sizes up to 10^8
    if avg_total_cpu_time<avg_total_cpu_loop_time:
        valid_main_methods =['CPU Add', 'add_device_mem_gpu', 'add_host_mem_gpu',
                        'add_gpuarray_kernel', 'add_gpuarray_no_kernel']
    else:
        valid_main_methods =['CPU_Loop_Add', 'add_device_mem_gpu', 'add_host_mem_gpu',
                        'add_gpuarray_kernel', 'add_gpuarray_no_kernel']
    valid_vector_sizes = vector_sizes

    for current_operation in valid_operations:
        # Initialize arrays to store the average times for each method
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        arr_avg_total_add_device_mem_gpu_time_incl = np.array([])
        arr_avg_total_add_device_mem_gpu_time_excl = np.array([])
        arr_avg_total_add_host_mem_gpu_time = np.array([])
        arr_avg_total_add_gpuarray_kernel_time_incl = np.array([])
        arr_avg_total_add_gpuarray_kernel_time_excl = np.array([])
        arr_avg_total_add_gpuarray_no_kernel_time_incl = np.array([])
        arr_avg_total_add_gpuarray_no_kernel_time_excl = np.array([])
        # Loop over the valid vector sizes
        for vector_size in valid_vector_sizes:
            # Convert the vector_size to an integer
            vector_size=int(vector_size)
            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            arr_total_add_device_mem_gpu_time_incl = np.array([])
            arr_total_add_device_mem_gpu_time_excl = np.array([])
            arr_total_add_host_mem_gpu_time = np.array([])
            arr_total_add_gpuarray_kernel_time_incl = np.array([])
            arr_total_add_gpuarray_kernel_time_excl = np.array([])
            arr_total_add_gpuarray_no_kernel_time_incl = np.array([])
            arr_total_add_gpuarray_no_kernel_time_excl = np.array([])

            print("vectorlength")
            print(vector_size)

            a_array_np = np.arange(1, vector_size + 1).astype(np.float32)
            b = 3  # Choose any number you desire
            b_number_np = np.float32(b)  # storing as number having value b with datatype Float32
            b_array_np = b * np.ones(vector_size).astype(np.float32)  # storing as array with all elements having equal value b as datatype Float32

            # Loop over the iterations
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if (current_operation == 'Pass Vector and Number'):
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if (current_method == 'CPU Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if (current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        elif (current_method == 'add_device_mem_gpu'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_device_mem_gpu(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_device_mem_gpu_time_incl = np.append(arr_total_add_device_mem_gpu_time_incl, total_time)
                            arr_total_add_device_mem_gpu_time_excl = np.append(arr_total_add_device_mem_gpu_time_excl, gpu_time)
                        elif (current_method == 'add_host_mem_gpu'):
                            c_gpu, total_time = graphicscomputer.add_host_mem_gpu(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_host_mem_gpu_time = np.append(arr_total_add_host_mem_gpu_time, total_time)
                        elif (current_method == 'add_gpuarray_kernel'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_gpuarray_kernel(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_gpuarray_kernel_time_incl = np.append(arr_total_add_gpuarray_kernel_time_incl, total_time)
                            arr_total_add_gpuarray_kernel_time_excl = np.append(arr_total_add_gpuarray_kernel_time_excl, gpu_time)
                        elif (current_method == 'add_gpuarray_no_kernel'):
                            c_gpu, total_time, gpu_time = graphicscomputer.add_gpuarray_no_kernel(a_array_np, b_in, vector_size, is_b_a_vector)
                            sum_diff = c_gpu - c_np_cpu_add
                            arr_total_add_gpuarray_no_kernel_time_incl = np.append(arr_total_add_gpuarray_no_kernel_time_incl, total_time)
                            arr_total_add_gpuarray_no_kernel_time_excl = np.append(arr_total_add_gpuarray_no_kernel_time_excl, gpu_time)
                        else:
                            print("Unknown method:", current_method)
                        total_diff = np.sum(np.abs(sum_diff))
                        if (total_diff != 0):
                            print(current_method + " " + current_operation + " sum mismatch")
                            print(total_diff)

            # Calculate the average times for each method
            avg_total_cpu_time = arr_total_cpu_time.mean()
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            # avg_total_cpu_loop_time = arr_total_cpu_loop_time.mean()
            # arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            avg_total_add_device_mem_gpu_time_incl = arr_total_add_device_mem_gpu_time_incl.mean()
            arr_avg_total_add_device_mem_gpu_time_incl = np.append(arr_avg_total_add_device_mem_gpu_time_incl, avg_total_add_device_mem_gpu_time_incl)
            avg_total_add_device_mem_gpu_time_excl = arr_total_add_device_mem_gpu_time_excl.mean()
            arr_avg_total_add_device_mem_gpu_time_excl = np.append(arr_avg_total_add_device_mem_gpu_time_excl, avg_total_add_device_mem_gpu_time_excl)
            avg_total_add_host_mem_gpu_time = arr_total_add_host_mem_gpu_time.mean()
            arr_avg_total_add_host_mem_gpu_time = np.append(arr_avg_total_add_host_mem_gpu_time, avg_total_add_host_mem_gpu_time)
            avg_total_add_gpuarray_kernel_time_incl = arr_total_add_gpuarray_kernel_time_incl.mean()
            arr_avg_total_add_gpuarray_kernel_time_incl = np.append(arr_avg_total_add_gpuarray_kernel_time_incl, avg_total_add_gpuarray_kernel_time_incl)
            avg_total_add_gpuarray_kernel_time_excl = arr_total_add_gpuarray_kernel_time_excl.mean()
            arr_avg_total_add_gpuarray_kernel_time_excl = np.append(arr_avg_total_add_gpuarray_kernel_time_excl, avg_total_add_gpuarray_kernel_time_excl)
            avg_total_add_gpuarray_no_kernel_time_incl = arr_total_add_gpuarray_no_kernel_time_incl.mean()
            arr_avg_total_add_gpuarray_no_kernel_time_incl = np.append(arr_avg_total_add_gpuarray_no_kernel_time_incl, avg_total_add_gpuarray_no_kernel_time_incl)
            avg_total_add_gpuarray_no_kernel_time_excl = arr_total_add_gpuarray_no_kernel_time_excl.mean()
            arr_avg_total_add_gpuarray_no_kernel_time_excl = np.append(arr_avg_total_add_gpuarray_no_kernel_time_excl, avg_total_add_gpuarray_no_kernel_time_excl)
        # Print the results
        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        # print(current_operation + " The CPU Loop times are")
        # print(arr_avg_total_cpu_loop_time)
        print(current_operation + " The add_device_mem_gpu times including memory transfer are")
        print(arr_avg_total_add_device_mem_gpu_time_incl)
        print(current_operation + " The add_device_mem_gpu times excluding memory transfer are")
        print(arr_avg_total_add_device_mem_gpu_time_excl)
        print(current_operation + " The add_host_mem_gpu times are")
        print(arr_avg_total_add_host_mem_gpu_time)
        print(current_operation + " The add_gpuarray_kernel times including memory transfer are")
        print(arr_avg_total_add_gpuarray_kernel_time_incl)
        print(current_operation + " The add_gpuarray_kernel times excluding memory transfer are")
        print(arr_avg_total_add_gpuarray_kernel_time_excl)
        print(current_operation + " The add_gpuarray_no_kernel times including memory transfer are")
        print(arr_avg_total_add_gpuarray_no_kernel_time_incl)
        print(current_operation + " The add_gpuarray_no_kernel times excluding memory transfer are")
        print(arr_avg_total_add_gpuarray_no_kernel_time_excl)

        # Plotting the results including memory transfer
        plt.figure(figsize=(10, 8))
        # plt.plot(valid_vector_sizes, arr_avg_total_cpu_time, label='CPU Add')
        # plt.plot(valid_vector_sizes, arr_avg_total_cpu_loop_time, label='CPU Loop Add')
        plt.plot(valid_vector_sizes, arr_avg_total_add_device_mem_gpu_time_incl, label='add_device_mem_gpu including memory transfer')
        plt.plot(valid_vector_sizes, arr_avg_total_add_host_mem_gpu_time, label='add_host_mem_gpu')
        plt.plot(valid_vector_sizes, arr_avg_total_add_gpuarray_kernel_time_incl, label='add_gpuarray_kernel including memory transfer')
        plt.plot(valid_vector_sizes, arr_avg_total_add_gpuarray_no_kernel_time_incl, label='add_gpuarray_no_kernel including memory transfer')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Vector Size (Power of 10)')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Average Execution Time Including Memory Transfer')
        plt.legend()
        plt.xticks(valid_vector_sizes, [str(int(np.log10(x))) for x in valid_vector_sizes])
        plt.grid(True)
        plt.savefig("cuda1.png")

        # Plotting the results excluding memory transfer
        plt.figure(figsize=(10, 8))
        plt.plot(valid_vector_sizes, arr_avg_total_add_device_mem_gpu_time_excl, label='add_device_mem_gpu excluding memory transfer')
        plt.plot(valid_vector_sizes, arr_avg_total_add_gpuarray_kernel_time_excl, label='add_gpuarray_kernel excl')
        plt.plot(valid_vector_sizes, arr_avg_total_add_gpuarray_no_kernel_time_excl, label='add_gpuarray_no_kernel excl')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Vector Size (Power of 10)')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Average Execution Time Excluding Memory Transfer')
        plt.legend()
        plt.xticks(valid_vector_sizes, [str(int(np.log10(x))) for x in valid_vector_sizes])
        plt.grid(True)
        plt.savefig("cuda2.png")