#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

struct AddFunctor {
    inline float initial() { return static_cast<float>(0.0f); }

    __device__ __forceinline__ float operator()(const float a, const float b) const {
        return b + a;
    }
};

__device__ __forceinline__ float BlockXReduce(float val, ReduceOp reducer) {
    __syncthreads();
    __shared__ float shared[64];
    int block_dim_x = blockDim.x;
    if (blockDim.x > WARP_SIZE) {
        block_dim_x = blockDim.x / WARP_SIZE;
        int lane = threadIdx.x % WARP_SIZE;
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int wid = tid / WARP_SIZE;
        int bid = threadIdx.y;
        val = WarpReduce(val, reducer);
        if (lane == 0) {
            shared[wid] = val;
        }
        __syncthreads();
        val = shared[bid * block_dim_x + lane];
    }

    for (int stride = 1; stride < block_dim_x; stride <<= 1) {
        float temp = CudaShuffleDownSync(val, stride);
        val = reducer(val, temp);
    }
    if (threadIdx.x == 0) {
        shared[threadIdx.y] = val;
    }
    __syncthreads();
    return shared[threadIdx.y];
}

__device__ __forceinline__ void ReduceMeanAndVar(
        float* mean, float* var, float x_mean, float x_var, int size) {
    const int nc = blockIdx.x;
    x_mean = BlockXReduce<float, AddFunctor<float>>(x_mean, AddFunctor<float>());
    x_var = BlockXReduce<float, AddFunctor<float>>(x_var, AddFunctor<float>());
    __syncthreads();
    if (threadIdx.x == 0) {
        mean[nc] = static_cast<float>(x_mean / size);
        var[nc] = static_cast<float>(x_var / size);
    }
}

extern "C"
__global__ void ScalarGetMeanAndVar(const float* x, float* mean, float* var, int size) {
    int i = blockIdx.x;
    float x_mean = static_cast<float>(0);
    float x_var = static_cast<float>(0);
    for (int j = threadIdx.x; j < size; j += blockDim.x) {
        float val;
        val = x[i * size + j];
        x_mean += val;
        x_var += val * val;
    }
    ReduceMeanAndVar<float>(mean, var, x_mean, x_var, size);
}