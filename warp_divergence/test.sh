nvprof ./warpDivergenceTest_cuda 1024000
nvprof --metrics branch_efficiency ./warpDivergenceTest_cuda 1024000
nvprof ./warpDivergenceTest_cuda 4096000
nvprof --metrics branch_efficiency ./warpDivergenceTest_cuda 4096000
nvprof ./warpDivergenceTest_cuda 10240000
nvprof --metrics branch_efficiency ./warpDivergenceTest_cuda 10240000
nvprof ./warpDivergenceTest_cuda 40960000
nvprof --metrics branch_efficiency ./warpDivergenceTest_cuda 40960000
nvprof ./warpDivergenceTest_cuda 102400000
nvprof --metrics branch_efficiency ./warpDivergenceTest_cuda 102400000