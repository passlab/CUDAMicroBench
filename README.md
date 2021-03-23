
# CUDAMicroBench
Microbenchmarks to Assist CUDA Performance Programming
## Summary of the CUDAMicroBench microbenchmarks
<table>
   <tr>
      <td>Benchmark name</td>
      <td>Pattern of Performance Inefficiency</td>
      <td>Optimization techniques</td>
   </tr>
   <tr>
      <td colspan="3">Optimizing Kernels to Saturate the Massive Parallel Capability of GPUs</td>
   </tr>
   <tr>
      <td>WarpDivRedux</td>
      <td>Threads enter different branches when they encounter the control flow statement</td>
      <td>Change the algorithm: take the warp size as the step</td>
   </tr>
   <tr>
      <td>DynParallel</td>
      <td>Workloads that require the use of nested parallelism such as those using adaptive grids</td>
      <td>Use dynamic parallelism to allow the GPU to generate its own work</td>
   </tr>
   <tr>
      <td>Conkernels</td>
      <td>Launch multiple kernel instances on one GPU</td>
      <td>Use concurrent kernels technique</td>
   </tr>
   <tr>
      <td>TaskGraph</td>
      <td>Provide a more effective model for submitting work to the GPU</td>
      <td>Pre-define the task graph and run-repeatedly execution flow</td>
   </tr>
   <tr>
      <td colspan="3">Effectively Leveraging the Deep Memory Hierarchy Inside GPU to Maximize The Computing Capability of GPU for Kernel Execution</td>
   </tr>
   <tr>
      <td>Shmem</td>
      <td>The data need to be accessed serveral times</td>
      <td>Use shared memory to store the data which needs to be accessed repeatly</td>
   </tr>
   <tr>
      <td>CoMem</td>
      <td>Stride or random access of array across threads which have uncoaleasced memory access</td>
      <td>Consecutive memory access across threads</td>
   </tr>
   <tr>
      <td>MemAlign</td>
      <td>Mallocation has unaliged adress at the begninning</td>
      <td>Use aligned malloc</td>
   </tr>
   <tr>
      <td>GSOverlap</td>
      <td>Global-shared memory copy takes much time</td>
      <td>Use the new function memcpy_async in CUDA11 to acclerate the data transfer</td>
   </tr>
   <tr>
      <td>Shuffle</td>
      <td>The data exchange between threads</td>
      <td>Use shuffle to enable threads in the same warp directly share part of their results between registers</td>
   </tr>
   <tr>
      <td>BankRedux</td>
      <td>Two or more threads access different locations of the same bank</td>
      <td>Change the algorithm to avoid bank conflicts</td>
   </tr>
   <tr>
      <td colspan="3">Properly Arranging Data Movement Between CPU and GPU to Reduce the Performance Impact of Data Movement for the GPU Program</td>
   </tr>
   <tr>
      <td>HDOverlap</td>
      <td>Host-device memory copy takes much time</td>
      <td>Use cudaMemcpyAsync function to acclerate the data transfer</td>
   </tr>
   <tr>
      <td>ReadOnlyMem</td>
      <td>Large amount of read-only data</td>
      <td>Put read-only data in constant/texture memory to get a higher speed</td>
   </tr>
   <tr>
      <td>UniMem</td>
      <td>Low memory access density</td>
      <td>Put the data in unified memory and only copy the necessary pages</td>
   </tr>
   <tr>
      <td>MiniTransfer</td>
      <td>Wrong data layout causes a large amount of useless data transfer between CPU and GPU</td>
      <td>Change to the correct data layout to avoid useless data transfer</td>
   </tr>
</table>

## Features
 1. A collection of fourteen microbenchmarks,
 1. Exhibit performance challenges in CUDA programming,
 1. Exhibit techniques to optimize the CUDA program to address the challenges,
 1. Includes examples and techniques for using advanced CUDA features,
 1. Help users to optimize CUDA program for performance

## Prerequisite
To execute microBenchmark, NVIDIA GPU and CUDA are needed. Some new features such as dynamic parallelism and memcpy_async require the GPU to support CUDA11.
    
## Experiment
There are fourteen examples so far you can experiment. You can check the Makefile in each folder to see how they can be compiled and then excute it by using the .sh file.

## Common folder

Several benchmarks are derived from CUDA Samples, including GSOverlap, ConKernels and Taskgraph. Some header files required by these three benchmarks are stored in the common folder, which also comes from CUDA Samples.

## [3-clause BSD License](LICENSE_BSD.txt) and Acknowledgement
Copyright (c) 2016 - 2019 Yonghong Yan and PASSlab (https://passlab.github.io) 
from University of South Carolina. All rights reserved. Funding for this research and 
development was provided by the National Science Foundation 
under award number CISE CCF-1833312, and CISE SHF-1551182. 



