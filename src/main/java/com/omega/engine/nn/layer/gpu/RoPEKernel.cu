#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

extern "C"
static __global__ void rope_norm(const float* x, float* dst,float* c_cos,float* c_sin, int ncols) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
	
    float cos_theta = c_cos[col / 2];
    float sin_theta = c_sin[col / 2];

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}