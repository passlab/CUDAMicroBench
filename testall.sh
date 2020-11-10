# test for coalesced and uncoalesced memory access
cd AXPY
make
sh test.sh >> results.txt

cd ../SpMM
make
nvprof ./SpMM_cuda >> results.txt

# test for Memory access density

cd ../LowAccessDensity
make
sh test.sh >> results.txt

# test for Use of read-only memory

cd ../AXPY_texture
make
sh test.sh >> results.txt

cd ../MatAdd2D_texture
make
sh test.sh >> results.txt

# test for Memory alignment

cd ../AXPY_Misaligned
make
sh test.sh >> results.txt

# test for Bank conflicts

cd ../Reduction_bank_conflicts
make
sh test.sh >> results.txt

# test Unnecessary data transfer

cd ../SpMV
make
sh test.sh >> results.txt

# test Warp divergence

cd ../warp_divergence
make
sh test.sh >> results.txt
