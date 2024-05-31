#define BLOCK 1024

extern "C"
__global__ void norm(float *X, float *Y, int N)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i == 0){
    	Y[0] = 0.0f;
    }
    if(i < 1) {
	    for(int index = 0;index<N;index++){
	    	Y[0] += powf(X[index], 2);
	    }
    }
    __syncthreads();
    if(i == 0){
    	Y[0] = sqrtf(Y[0]);
    }
}

extern "C"
__global__ void l2NormKernel(const float* d_vector, float* d_partialSum, int size) {
    extern __shared__ float sharedData[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sharedData[tid] = (i < size) ? d_vector[i] * d_vector[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) d_partialSum[blockIdx.x] = sharedData[0];
}