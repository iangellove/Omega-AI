#define BLOCK 1024 


extern "C"
__global__ void relu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		output[i] = x[i];
    	}else{
    		output[i] = 0;
    	}
    }
}

extern "C"
__global__ void relu_backward(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] = delta[i];
    	}else{
    		diff[i] = 0;
    	}
    }
}

extern "C"
__global__ void leakyRelu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		output[i] = x[i];
    	}else{
    		output[i] = x[i] * 0.2f;
    	}
    }
}

extern "C"
__global__ void leakyRelu_backward(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] = delta[i];
    	}else{
    		diff[i] = delta[i] * 0.2f;
    	}
    }
}

extern "C"
__global__ void sigmod_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = (float) (1.0f / (1.0f + expf(-x[i])));
    }
}

extern "C"
__global__ void sigmod_backward(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] = delta[i] * output[i] * (1.0f - output[i]);
    }
}
