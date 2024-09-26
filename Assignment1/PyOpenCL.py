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

import clModule
import numpy as np
from clModule import clModule
import matplotlib.pyplot as plt
# import relevant libraries


if __name__ == "__main__":
    # List all main methods
    all_main_methods = ['CPU_numpy_Add', 'CPU_Loop_Add', 'DeviceAdd', 'BufferAdd']
    # List the two operations
    all_operations = ['Pass Vector and Number', 'Pass Two Vectors']
    # List the size of vectors
    vector_sizes = 10**np.arange(1,9)
    # List iteration indexes
    iteration_indexes = np.arange(1,50)

    # Select the list of valid methods to perform (populate as you complete the methods).
    # Currently in template code only CPU Add and CPU Loop Add are complete.
    valid_main_methods = all_main_methods[0:2]

    # Select the list of valid operations to be run
    valid_operations = all_operations

    # Select the list of valid vector_sizes for current_analysis
    valid_vector_sizes = vector_sizes[0:6]

    # Create an instance of the clModule class
    graphicscomputer = clModule()

    # Nested loop precedence, operations -> vector_size -> iteration -> CPU/GPU method.
    # There are four nested loops, the main loop iterates between performing vector + number, and performing vector + vector cases.
    # The second loop iterates between different vector sizes, for each case of the main loop.
    # The third loop runs 50 repetitions, for each case of the second loop
    # The fourth loop iterates between the different CPU/GPU/Memory-transfer methods, for each case of the third loop.

    for current_operation in valid_operations:
        # Set initial arrays to populate average computation times for different vector sizes
        arr_avg_total_cpu_time = np.array([])
        arr_avg_total_cpu_loop_time = np.array([])
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        arr_avg_total_device_time = np.array([])
        arr_avg_total_buffer_time = np.array([])
        #arr_avg_total_device_no_transfer_time = np.array([])
        #arr_avg_total_buffer_no_transfer_time = np.array([])

        
        for vector_size in valid_vector_sizes:

            arr_total_cpu_time = np.array([])
            arr_total_cpu_loop_time = np.array([])

            # [TODO: Students should write Code]
            # Add for the rest of the methods
            arr_total_device_time = np.array([])
            arr_total_buffer_time = np.array([])
            arr_total_device_no_transfer_time = np.array([])
            arr_total_buffer_no_transfer_time = np.array([])

            print ("vectorlength")
            print (vector_size)

            a_array_np = np.arange(1,vector_size+1).astype(np.float32) # Generating a vector having values 1 to vector_size as Float32 datatype.
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
                    if(current_method == 'CPU_numpy_Add'):
                        c_np_cpu_add, cpu_time_add = graphicscomputer.CPU_numpy_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                        arr_total_cpu_time = np.append(arr_total_cpu_time, cpu_time_add)
                    else:
                        if(current_method == 'CPU_Loop_Add'):
                            c_np_cpu_loop_add, cpu_time_loop_add = graphicscomputer.CPU_Loop_Add(a_array_np,b_in,vector_size,is_b_a_vector)
                            sum_diff = c_np_cpu_loop_add - c_np_cpu_add
                            arr_total_cpu_loop_time = np.append(arr_total_cpu_loop_time, cpu_time_loop_add)
                        
                        # [TODO: Students should write Code]
                        # Add for the rest of the methods

                        if (current_method == 'DeviceAdd'):
                            c_np_device_add, device_time_add = graphicscomputer.deviceAdd(a_array_np,b_in,vector_size,is_b_a_vector)
                            arr_total_device_time = np.append(arr_total_device_time, device_time_add)
                            sum_diff = c_np_device_add - c_np_cpu_add

                        if (current_method == 'BufferAdd'):
                            c_np_buffer_add, buffer_time_add = graphicscomputer.bufferAdd(a_array_np,b_in,vector_size,is_b_a_vector)
                            arr_total_buffer_time = np.append(arr_total_buffer_time, buffer_time_add)
                            sum_diff = c_np_buffer_add - c_np_cpu_add
                                    




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
            avg_total_device_time = ((arr_total_device_time.sum())/50)
            arr_avg_total_device_time = np.append(arr_avg_total_device_time, avg_total_device_time)
            avg_total_buffer_time = ((arr_total_buffer_time.sum())/50)
            arr_avg_total_buffer_time = np.append(arr_avg_total_buffer_time, avg_total_buffer_time)

        print(current_operation + " The CPU times are")
        print(arr_avg_total_cpu_time)
        print(current_operation + " The CPU Loop times are")
        print(arr_avg_total_cpu_loop_time)
        # [TODO: Students should write Code]
        # Add for the rest of the methods
        print(current_operation + " The Device times are")
        print(arr_avg_total_device_time)
        print(current_operation + " The Buffer times are")
        print(arr_avg_total_buffer_time)


        # Code for Plotting the results
        plt.figure()
        plt.loglog(valid_vector_sizes, arr_avg_total_cpu_time, label='CPU numpy Add')
        plt.loglog(valid_vector_sizes, arr_avg_total_device_time, label='DeviceAdd')
        plt.loglog(valid_vector_sizes, arr_avg_total_buffer_time, label='BufferAdd')
        #plt.loglog(valid_vector_sizes, arr_avg_total_device_no_transfer_time, label='DeviceAddNoTransfer')
        #plt.loglog(valid_vector_sizes, arr_avg_total_buffer_no_transfer_time, label='BufferAddNoTransfer')

        plt.xlabel('Vector Size')
        plt.ylabel('Average Execution Time (s)')
        plt.title(f'Execution Time vs Vector Size ({current_operation})')
        plt.legend()
        plt.xticks(valid_vector_sizes, [f'$10^{int(np.log10(size))}$' for size in valid_vector_sizes])
        plt.show() 
