#define C10_WARP_SIZE 32
constexpr int kCUDABlockReduceNumThreads = 512;
constexpr int kCUDABlockReduceMaxThreads = C10_WARP_SIZE * C10_WARP_SIZE;

extern "C"
__device__ __forceinline__ float WARP_SHFL_DOWN(float value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff){
#if !defined(USE_ROCM)
    return __shfl_down_sync(mask, value, delta, width);
#else
    return __shfl_down(value, delta, width);
#endif
}

extern "C"
__inline__ __device__ float WarpReduceSum(float val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

__inline__ __device__ float BlockReduceSum(float val, float* shared) {
  const int tid = threadIdx.x;
  const int lid = tid % C10_WARP_SIZE;
  const int wid = tid / C10_WARP_SIZE;
  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < blockDim.x / C10_WARP_SIZE) ? shared[lid] : float(0);
  if (wid == 0) {
    val = WarpReduceSum(val);
  }
  return val;
}

extern "C"
__device__ __inline__ void compute_gI(
  const float* __restrict__ dY,
  const float* __restrict__ X,
  const float* __restrict__ mean,
  const float* __restrict__ rstd,
  const float* __restrict__ gamma,
  float* dX,
  const int N,
  float * buf){
    const auto i1 = blockIdx.x;
    const float mean_val = mean[i1];
    const float rstd_val = rstd[i1];
    float stats_x1{0}, stats_x2{0};
    constexpr int unroll = 4;
    auto l = unroll * threadIdx.x;
    const float* X_i = X + i1 * N;
    const float* dY_i = dY + i1 * N;
    float* dX_i = dX + i1 * N;
    //vectorized reads don't improve perf, so use regular unrolling

    for (; l+unroll - 1 < N; l += blockDim.x * unroll){
      #pragma unroll
      for (int k=0; k< unroll; k++){
          const auto gamma_val = (gamma != nullptr) ? static_cast<float>(gamma[l+k]) : float(1);
          const auto c_h = static_cast<float>(X_i[l+k]);
          const auto c_loss = static_cast<float>(dY_i[l+k]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
      }
    }
    for (;  l < N; l ++) {
          const auto gamma_val = (gamma != nullptr) ? static_cast<float>(gamma[l]) : float(1);
          const auto c_h = static_cast<float>(X_i[l]);
          const auto c_loss = static_cast<float>(dY_i[l]);
          stats_x1 += c_loss * gamma_val;
          stats_x2 += c_loss * gamma_val * (c_h - mean_val) * rstd_val;
    }

    stats_x1 = BlockReduceSum(stats_x1, buf);
    stats_x2 = BlockReduceSum(stats_x2, buf);
    if (threadIdx.x == 0) {
      buf[0] = stats_x1;
      buf[1] = stats_x2;
    }
    __syncthreads();
    stats_x1 = buf[0];
    stats_x2 = buf[1];
    float fH = N;
    float term1 = (float(1) / fH) * rstd_val;

    for (int l = threadIdx.x; l < N; l += blockDim.x){
        const auto x = X_i[l];
        const auto dy = dY_i[l];
        const auto gamma_val = (gamma != nullptr) ? static_cast<float>(gamma[l]) : float(1);

        float f_grad_input = fH * gamma_val * dy;
        f_grad_input -= (x - mean_val) * rstd_val * stats_x2;
        f_grad_input -= stats_x1;
        f_grad_input *= term1;
        dX_i[l] = f_grad_input;
    }
  }
  
extern "C"
__global__ void aten_layer_norm_grad_input_kernel(
  const float* __restrict__ dY,
  const float* __restrict__ X,
  const float* __restrict__ mean,
  const float* __restrict__ rstd,
  const float* __restrict__ gamma,
  float*  dX,
  const int N){
    alignas(sizeof(double)) extern __shared__ char s_data1[];
    float * buf = reinterpret_cast<float*>(&s_data1);

    compute_gI(dY, X, mean, rstd, gamma, dX, N, buf);
  }  