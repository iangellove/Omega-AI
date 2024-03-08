#define BLOCK 1024 


extern "C"
__device__ __inline__ float get_value(const float* index, const int bound_check,
                                  const int up_bound) {
  if (bound_check < up_bound)
    return __ldg(index);
  else
    return 0.0f;
}

extern "C"
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

extern "C"
__global__ void LayerNormFusedSmallGPUKernel(
    const int slice_size,const int in_depth,const int n_inputs,const float epsilon, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ output, const int num_blocks, const int slice_per_block) {

  const float i_n = 1.0f / in_depth;

  const int slice_id = threadIdx.x / slice_size;
  const int tSliceIdx = threadIdx.x % slice_size;

  const float _gamma = get_value(gamma + tSliceIdx, tSliceIdx, in_depth);
  const float _beta = get_value(beta + tSliceIdx, tSliceIdx, in_depth);

  float mu;
  float rstd;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
		mu = 0.0f;
		rstd = 0.0f;
		
		const int thread_id = (bId * slice_per_block + slice_id) * in_depth + tSliceIdx;
		// const T inp = 0;
		const float inp = get_value(input + thread_id, tSliceIdx, in_depth);
		
		mu += inp * i_n;
		
		for (int mask = slice_size / 2; mask > 0; mask /= 2) {
		  mu += __shfl_xor(mu, mask);
		}
		
		if (tSliceIdx < in_depth) rstd += (inp - mu) * (inp - mu);
		// rstd += (inp-mu)*(inp-mu);
		
		for (int mask = slice_size / 2; mask > 0; mask /= 2) {
		  rstd += __shfl_xor(rstd, mask);
		}
		
		rstd = rsqrt(rstd * i_n + epsilon);
		
		if (tSliceIdx < in_depth && thread_id < n_inputs)
		  output[thread_id] = (inp - mu) * rstd * _gamma + _beta;
  }
}

extern "C"
__global__ void LayerNorm1GPUKernel(const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const int tSliceIdx = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  const int mult = 1;
  const float i_n = 1.0f / in_depth;
  float inp[mult];
  float _gamma[mult];
  float _beta[mult];
  int thread_id[mult];

  float sum;
  float sqSum;
  float mu;
  float rstd;
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _gamma[m] = get_value(gamma + tSliceIdx + m * blockDim.x,
                             tSliceIdx + m * blockDim.x, in_depth);
    _beta[m] = get_value(beta + tSliceIdx + m * blockDim.x,
                            tSliceIdx + m * blockDim.x, in_depth);
  }
  
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = 0.0f;
    sqSum = 0.0f;

    if (tSliceIdx == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
    }
    __syncthreads();

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + tSliceIdx;
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], tSliceIdx + m * blockDim.x,
                            in_depth);
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) { sum += inp[m] * i_n; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], sum);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], sqSum);
    }
    __syncthreads();
    if (tSliceIdx == 0) {
      std_cache[0] = rsqrt(std_cache[0] * i_n + epsilon);
    }
    __syncthreads();
    rstd = std_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
         output[thread_id[m]] = (inp[m] - mu) * rstd * _gamma[m] + _beta[m];
      }
    }
    __syncthreads();
  }
}

extern "C"
__global__ void LayerNorm2GPUKernel(const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const int tSliceIdx = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  const int mult = 2;
  const float i_n = 1.0f / in_depth;
  float inp[mult];
  float _gamma[mult];
  float _beta[mult];
  int thread_id[mult];

  float sum;
  float sqSum;
  float mu;
  float rstd;
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _gamma[m] = get_value(gamma + tSliceIdx + m * blockDim.x,
                             tSliceIdx + m * blockDim.x, in_depth);
    _beta[m] = get_value(beta + tSliceIdx + m * blockDim.x,
                            tSliceIdx + m * blockDim.x, in_depth);
  }
  
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = 0.0f;
    sqSum = 0.0f;

    if (tSliceIdx == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
    }
    __syncthreads();

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + tSliceIdx;
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], tSliceIdx + m * blockDim.x,
                            in_depth);
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) { sum += inp[m] * i_n; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], sum);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], sqSum);
    }
    __syncthreads();
    if (tSliceIdx == 0) {
      std_cache[0] = rsqrt(std_cache[0] * i_n + epsilon);
    }
    __syncthreads();
    rstd = std_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
         output[thread_id[m]] = (inp[m] - mu) * rstd * _gamma[m] + _beta[m];
      }
    }
    __syncthreads();
  }
}

extern "C"
__global__ void LayerNorm3GPUKernel(const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const int tSliceIdx = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  const int mult = 3;
  const float i_n = 1.0f / in_depth;
  float inp[mult];
  float _gamma[mult];
  float _beta[mult];
  int thread_id[mult];

  float sum;
  float sqSum;
  float mu;
  float rstd;
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _gamma[m] = get_value(gamma + tSliceIdx + m * blockDim.x,
                             tSliceIdx + m * blockDim.x, in_depth);
    _beta[m] = get_value(beta + tSliceIdx + m * blockDim.x,
                            tSliceIdx + m * blockDim.x, in_depth);
  }
  
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = 0.0f;
    sqSum = 0.0f;

    if (tSliceIdx == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
    }
    __syncthreads();

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + tSliceIdx;
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], tSliceIdx + m * blockDim.x,
                            in_depth);
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) { sum += inp[m] * i_n; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], sum);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], sqSum);
    }
    __syncthreads();
    if (tSliceIdx == 0) {
      std_cache[0] = rsqrt(std_cache[0] * i_n + epsilon);
    }
    __syncthreads();
    rstd = std_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
         output[thread_id[m]] = (inp[m] - mu) * rstd * _gamma[m] + _beta[m];
      }
    }
    __syncthreads();
  }
}

extern "C"
__global__ void LayerNorm4GPUKernel(const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const int tSliceIdx = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  const int mult = 4;
  const float i_n = 1.0f / in_depth;
  float inp[mult];
  float _gamma[mult];
  float _beta[mult];
  int thread_id[mult];

  float sum;
  float sqSum;
  float mu;
  float rstd;
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _gamma[m] = get_value(gamma + tSliceIdx + m * blockDim.x,
                             tSliceIdx + m * blockDim.x, in_depth);
    _beta[m] = get_value(beta + tSliceIdx + m * blockDim.x,
                            tSliceIdx + m * blockDim.x, in_depth);
  }
  
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = 0.0f;
    sqSum = 0.0f;

    if (tSliceIdx == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
    }
    __syncthreads();

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + tSliceIdx;
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], tSliceIdx + m * blockDim.x,
                            in_depth);
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) { sum += inp[m] * i_n; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], sum);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], sqSum);
    }
    __syncthreads();
    if (tSliceIdx == 0) {
      std_cache[0] = rsqrt(std_cache[0] * i_n + epsilon);
    }
    __syncthreads();
    rstd = std_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
         output[thread_id[m]] = (inp[m] - mu) * rstd * _gamma[m] + _beta[m];
      }
    }
    __syncthreads();
  }
}

extern "C"
__global__ void LayerNorm5GPUKernel(const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks) {

  const int tWarpIdx = threadIdx.x % warpSize;

  const int tSliceIdx = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(float)) unsigned char my_smem[];
  float* mean_cache = (float*)my_smem;
  float* std_cache = (float*)&my_smem[sizeof(float)];
  const int mult = 5;
  const float i_n = 1.0f / in_depth;
  float inp[mult];
  float _gamma[mult];
  float _beta[mult];
  int thread_id[mult];

  float sum;
  float sqSum;
  float mu;
  float rstd;
  
  #pragma unroll
  for (int m = 0; m < mult; m++) {
    _gamma[m] = get_value(gamma + tSliceIdx + m * blockDim.x,
                             tSliceIdx + m * blockDim.x, in_depth);
    _beta[m] = get_value(beta + tSliceIdx + m * blockDim.x,
                            tSliceIdx + m * blockDim.x, in_depth);
  }
  
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = 0.0f;
    sqSum = 0.0f;

    if (tSliceIdx == 0) {
      mean_cache[0] = 0.0f;
      std_cache[0] = 0.0f;
    }
    __syncthreads();

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + tSliceIdx;
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) {
      inp[m] = get_value(input + thread_id[m], tSliceIdx + m * blockDim.x,
                            in_depth);
    }
	
	#pragma unroll
    for (int m = 0; m < mult; m++) { sum += inp[m] * i_n; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&mean_cache[0], sum);
    }
    __syncthreads();

    mu = mean_cache[0];
    
    #pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (tWarpIdx == 0) {
      atomicAdd(&std_cache[0], sqSum);
    }
    __syncthreads();
    if (tSliceIdx == 0) {
      std_cache[0] = rsqrt(std_cache[0] * i_n + epsilon);
    }
    __syncthreads();
    rstd = std_cache[0];

	#pragma unroll
    for (int m = 0; m < mult; m++) {
      if (tSliceIdx + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
         output[thread_id[m]] = (inp[m] - mu) * rstd * _gamma[m] + _beta[m];
      }
    }
    __syncthreads();
  }
}


