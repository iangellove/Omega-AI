#define BLOCK 1024

extern "C"
__global__ void norm(int N, float *X, float *Y)
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