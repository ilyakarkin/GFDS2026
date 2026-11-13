# cuDSS 1.0.0 (CUDA 12) for the UGLMAT bench: . ~/cudss/env.sh
# Build: . ~/nv.sh; . ./env.sh; nvc++ -O2 -std=c++17 -cuda -gpu=cc80,cuda12.9 -I$CUDSS_ROOT/include uglb_cudss.cu -o uglb_cudss -L$CUDSS_ROOT/lib -lcudss
N=/opt/nvidia/hpc_sdk/Linux_x86_64/26.9
export CUDSS_ROOT=$HOME/cudss/libcudss-linux-x86_64-1.0.0.35_cuda12-archive
export LD_LIBRARY_PATH=$CUDSS_ROOT/lib:$N/math_libs/12.9/lib64:$N/cuda/12.9/targets/x86_64-linux/lib:$N/compilers/lib:$LD_LIBRARY_PATH
