#define BLOCK 1024 
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

__device__ void atomicAddX(float* addr, float val) {
    atomicAdd(addr, val);
}

extern "C"
__global__ void rmsnorm_forward_kernel(float* __restrict__ out, float* __restrict__ smean, float* __restrict__ rms,const float*  __restrict__ weight,
                                    const float*  __restrict__ inp, int N, int C) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    // thread coarsening through the row, reduce the sum in series
    float thread_sum2 = 0.0; // stores sum(x**2)
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float xi = x[i];
        thread_sum2 += xi * xi;
    }
    // warp-level reduction
    float warp_sum2 = cg::reduce(warp, thread_sum2, cg::plus<float>{}); // sum(x**2)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x**2)
    // mean
    block_sum2 /= C; // mean(x**2)
    float sm = block_sum2;
    float rsqrt = rsqrtf(block_sum2 + 1e-5f);
    // store the mean, no need to cache it
    if(threadIdx.x == 0 && smean != nullptr) {
        __stcs(smean + idx, sm);
    }
    if(threadIdx.x == 0 && rms != nullptr) {
        __stcs(rms + idx, rsqrt);
    }
    // final normalization and scaling by weight/bias
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = x[i] * rsqrt;
        __stcs(o+i, n * weight[i]);
    }
}

extern "C"
__global__ void rmsnorm_backward_kernel(float* __restrict__ out, float* __restrict__ dweight, float* __restrict__ smean, float* __restrict__ rms,
 const float*  __restrict__ inp, const float*  __restrict__ delta, const float* __restrict__ weight, int N, int C) {
 	extern __shared__ float shared[]; // size = C
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float shared_sum2[32]; // warps will be writing into shared memeory after warp-reduce
    int num_warps = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int idx = blockIdx.x; // simpoy one block per row
    
    float* dweight_shared = shared;
    
    // the row of input that this group of threads is responsible for
    const float* x = inp + idx * C;
    const float* d = delta + idx * C;
    
     // init shared memory to zero
    #pragma unroll 4
    for(int i = threadIdx.x; i < C; i+= blockDim.x){
       dweight_shared[i] = 0.0f;
    }
    __syncthreads();
    
    float b = -0.5 * powf(smean[idx] + 1e-5f, -1.5);
    
    // thread coarsening through the row, reduce the sum in series
    float drms_sum = 0.0; // stores sum(x * d * weight)
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        drms_sum += x[i] * d[i] * weight[i];
    }
    // warp-level reduction
    float warp_sum2 = cg::reduce(warp, drms_sum, cg::plus<float>{}); // sum(x * d * weight)
    // store the warp-level reduction in shared memory (we could have lane_id == 0 guard but not needed)
    shared_sum2[warp_id] = warp_sum2;
    __syncthreads();
    // load results from shared memory to threads, pad with zeros for threads that are out of bounds
    warp_sum2 = (lane_id < num_warps) ? shared_sum2[lane_id] : 0.0f;
    // now reduce the warp-level reductions
    float block_sum2 = cg::reduce(warp, warp_sum2, cg::plus<float>{}); // sum(x * d * weight)
	
    float* o = out + idx * C;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float n = rms[idx] * d[i] * weight[i] + block_sum2 * b / C * 2 * x[i];
        dweight_shared[i] = x[i] * rms[idx] * d[i];
        __stcs(o+i, n);
    }
    
    __syncthreads();

    for(int i = threadIdx.x; i < C; i+= blockDim.x) {
        atomicAddX(&dweight[i], (float)dweight_shared[i]);
    }
}