#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

extern "C"
__global__ void rope_norm(const float* x, float* dst,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (ncols / headSize / 2) + col / 2 % headSize;
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

extern "C"
__global__ void rope_backward(float* delta, float* diff,float* c_cos,float* c_sin, int ncols, int T,int headSize) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);
    if (col >= ncols) {
        return;
    }
	
	const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int t = row % T;
	const int ai = t * (ncols / headSize / 2) + col / 2 % headSize;
	
    float cos_theta = c_cos[ai];
    float sin_theta = c_sin[ai];

    const float d0 = delta[i + 0];
    const float d1 = delta[i + 1];

    diff[i + 0] = d0*cos_theta + d1*sin_theta;
    diff[i + 1] = d1*cos_theta - d0*sin_theta;
}