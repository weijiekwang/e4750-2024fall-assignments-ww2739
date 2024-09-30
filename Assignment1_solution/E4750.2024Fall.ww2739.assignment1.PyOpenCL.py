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
import importlib.util
import sys
# import clModule
# clmodule=__import__('E4750.2024Fall.ww2739.assignment1.clModule')
# clModule=importlib.import_module('E4750.2024Fall.ww2739.assignment1.clModule')
# from clmodule import clModule
# from clModule import clModule
module_name = "E4750.2024Fall.ww2739.assignment1.clModule"
module_path = "/home/ww2739/E4750.2024Fall.ww2739.assignment1.clModule.py"

spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # List all main methods
    all_main_methods = ['CPU numpy Add', 'CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10 ** np.arange(1, 9)
    # List iteration indexes
    iteration_indexes = np.arange(1, 50)

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods  # Now including all methods

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid vector_sizes for current_analysis
    # Initially, up to 10^6
    valid_vector_sizes_initial = vector_sizes[0:6]

    # Create an instance of the clModule class
    graphicscomputer = module.clModule()

    # First part: Run all methods up to vector size 10^6
    for current_operation in valid_operations:
        # Set initial arrays to populate average computation times for different vector sizes
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        arr_avg_total_device_time = np.array([])
        arr_avg_total_buffer_time = np.array([])
        # Loop through verctor sizes up to 10^6
        for vector_size in valid_vector_sizes_initial:
            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])
            arr_total_device_time = np.array([])
            arr_total_buffer_time = np.array([])
            # Print vector size
            print("Vector Length:", vector_size)

            a_array_np = np.arange(1, vector_size + 1).astype(np.float32)
            b = 3  # Choose any number you desire
            b_number_np = np.float32(b)
            b_array_np = b * np.ones(vector_size).astype(np.float32)
            # Loop through iterations
            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if current_operation == 'Pass Vector and Number':
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np

                    if current_method == 'CPU numpy Add':
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_numpy_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    elif current_method == 'CPU_Loop_Add':
                        c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                        sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                        arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        total_diff = sum_diff.sum()
                        if total_diff != 0:
                            print(f"{current_method} {current_operation} sum mismatch")
                            print(total_diff)
                    elif current_method == 'DeviceAdd':
                        c_device_add, device_time = graphicscomputer.deviceAdd(a_array_np, b_in, vector_size, is_b_a_vector)
                        sum_diff = c_device_add - c_np_cpu_add
                        arr_total_device_time = np.append(arr_total_device_time, device_time)
                        total_diff = sum_diff.sum()
                        if total_diff != 0:
                            print(f"{current_method} {current_operation} sum mismatch")
                            print(total_diff)
                        
                    elif current_method == 'BufferAdd':
                        c_buffer_add, buffer_time = graphicscomputer.bufferAdd(a_array_np, b_in, vector_size, is_b_a_vector)
                        sum_diff = c_buffer_add - c_np_cpu_add
                        arr_total_buffer_time = np.append(arr_total_buffer_time, buffer_time)
                        total_diff = sum_diff.sum()
                        if total_diff != 0:
                            print(f"{current_method} {current_operation} sum mismatch")
                            print(total_diff)

            # Compute average times
            avg_total_cpu_time = arr_total_cpu_time.mean()
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_cpu_loop_time = arr_total_cpu_loop_time.mean()
            arr_avg_total_cpu_loop_time = np.append(arr_avg_total_cpu_loop_time, avg_total_cpu_loop_time)
            avg_total_device_time = arr_total_device_time.mean()
            arr_avg_total_device_time = np.append(arr_avg_total_device_time, avg_total_device_time)
            avg_total_buffer_time = arr_total_buffer_time.mean()
            arr_avg_total_buffer_time = np.append(arr_avg_total_buffer_time, avg_total_buffer_time)

        # Print average times
        print(f"{current_operation} - CPU numpy Add times:")
        print(arr_avg_total_cpu_time)
        print(f"{current_operation} - CPU Loop Add times:")
        print(arr_avg_total_cpu_loop_time)
        print(f"{current_operation} - DeviceAdd times:")
        print(arr_avg_total_device_time)

        device_scalar=arr_avg_total_device_time
        device_scalar=np.append(device_scalar,0)
        device_scalar=np.append(device_scalar,0)

        print(f"{current_operation} - BufferAdd times:")
        print(arr_avg_total_buffer_time)
        buffer_scalar=arr_avg_total_buffer_time
        buffer_scalar=np.append(buffer_scalar,0)
        buffer_scalar=np.append(buffer_scalar,0)
    
        # Plotting results for initial vector sizes
        vector_sizes_plot = valid_vector_sizes_initial
        plt.figure(figsize=(10, 6))
        plt.loglog(vector_sizes_plot, arr_avg_total_cpu_time, label='CPU numpy Add')
        plt.loglog(vector_sizes_plot, arr_avg_total_cpu_loop_time, label='CPU Loop Add')
        plt.loglog(vector_sizes_plot, arr_avg_total_device_time, label='DeviceAdd (Add_to_each_element_GPU)')
        plt.loglog(vector_sizes_plot, arr_avg_total_buffer_time, label='BufferAdd (Add_to_each_element_GPU)')
        plt.xlabel('Vector Size (N)')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title(f'Performance Comparison ({current_operation}) - Up to 10^6')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xticks(vector_sizes_plot, labels=[f'$10^{int(np.log10(size))}$' for size in vector_sizes_plot])
        plt.savefig("opencl1.png")


    # Second part: Exclude slower CPU method and extend vector sizes up to 10^8
    if (arr_avg_total_cpu_time.mean()<arr_avg_total_cpu_loop_time.mean()):
        valid_main_methods = ['CPU numpy Add', 'DeviceAdd', 'BufferAdd']
    else:
        valid_main_methods = ['CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # Extend vector sizes up to 10^8
    valid_vector_sizes_extended = vector_sizes

    for current_operation in valid_operations:
        # Set initial arrays to populate average computation times for different vector sizes
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_device_time = np.array([])
        arr_avg_total_buffer_time = np.array([])
        # Loop through verctor sizes up to 10^8
        for vector_size in valid_vector_sizes_extended:
            arr_total_cpu_time = np.array([])
            arr_total_device_time = np.array([])
            arr_total_buffer_time = np.array([])

            print("Vector Length:",vector_size)

            a_array_np = np.arange(1, vector_size + 1).astype(np.float32)
            b = 3  # Choose any number you desire
            b_number_np = np.float32(b)
            b_array_np = b * np.ones(vector_size).astype(np.float32)

            for iteration in iteration_indexes:
                for current_method in valid_main_methods:
                    if current_operation == 'Pass Vector and Number':
                        is_b_a_vector = False
                        b_in = b_number_np
                    else:
                        is_b_a_vector = True
                        b_in = b_array_np
                    if current_method == 'CPU numpy Add':
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_numpy_Add(a_array_np, b_in, vector_size, is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    elif current_method == 'DeviceAdd':
                        c_device_add, device_time = graphicscomputer.deviceAdd(a_array_np, b_in, vector_size, is_b_a_vector)
                        sum_diff = c_device_add - c_np_cpu_add
                        arr_total_device_time = np.append(arr_total_device_time, device_time)
                    elif current_method == 'BufferAdd':
                        c_buffer_add, buffer_time = graphicscomputer.bufferAdd(a_array_np, b_in, vector_size, is_b_a_vector)
                        sum_diff = c_buffer_add - c_np_cpu_add
                        arr_total_buffer_time = np.append(arr_total_buffer_time, buffer_time)
                    total_diff = sum_diff.sum()
                    if total_diff != 0:
                        print(f"{current_method} {current_operation} sum mismatch")
                        print(total_diff)

            # Compute average times
            avg_total_cpu_time = arr_total_cpu_time.mean()
            arr_avg_total_cpu_time = np.append(arr_avg_total_cpu_time, avg_total_cpu_time)
            avg_total_device_time = arr_total_device_time.mean()
            arr_avg_total_device_time = np.append(arr_avg_total_device_time, avg_total_device_time)
            avg_total_buffer_time = arr_total_buffer_time.mean()
            arr_avg_total_buffer_time = np.append(arr_avg_total_buffer_time, avg_total_buffer_time)

        # Print average times
        print(f"{current_operation} The CPU Times are:")
        print(arr_avg_total_cpu_time)
        print(f"{current_operation} The Device Add Times are:")
        print(arr_avg_total_device_time)
        print(f"{current_operation} The Buffer Add Times are:")
        print(arr_avg_total_buffer_time)

        # Plotting results for extended vector sizes
        vector_sizes_plot = valid_vector_sizes_extended
        plt.figure(figsize=(10, 8))
        plt.loglog(vector_sizes_plot, arr_avg_total_cpu_time, label='CPU numpy Add')
        plt.loglog(vector_sizes_plot, arr_avg_total_device_time, label='DeviceAdd (Add Two Vectors GPU)')
        plt.loglog(vector_sizes_plot, arr_avg_total_buffer_time, label='BufferAdd (Add Two Vectors GPU)')
        # plt.loglog(vector_sizes_plot, device_scalar, label='DeviceAdd (Add_to_each_element_GPU)')
        # plt.loglog(vector_sizes_plot, buffer_scalar, label='BufferAdd (Add_to_each_element_GPU)')
        plt.xlabel('Vector Size (N)')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title(f'Performance Comparison ({current_operation}) - Up to 10^8')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xticks(vector_sizes_plot, labels=[f'$10^{int(np.log10(size))}$' for size in vector_sizes_plot])
        plt.savefig("opencl2.png")