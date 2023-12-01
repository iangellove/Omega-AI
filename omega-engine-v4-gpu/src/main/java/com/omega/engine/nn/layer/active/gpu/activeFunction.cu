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
__global__ void relu_backward_temp(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] += delta[i];
    	}else{
    		diff[i] += 0;
    	}
    }
}

extern "C"
__global__ void leakyRelu_forward(float *x, float *output, int n,float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		output[i] = x[i];
    	}else{
    		output[i] = x[i] * s;
    	}
    }
}

extern "C"
__global__ void leakyRelu_backward(float *x, float *delta, float *diff, int n,float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] = delta[i];
    	}else{
    		diff[i] = delta[i] * s;
    	}
    }
}

extern "C"
__global__ void leakyRelu_backward_temp(float *x, float *delta, float *diff, int n, float s)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	if(x[i] > 0){
    		diff[i] += delta[i];
    	}else{
    		diff[i] += delta[i] * s;
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

extern "C"
__global__ void sigmod_backward_temp(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * output[i] * (1.0f - output[i]);
    }
}

extern "C"
__global__ void tanh_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	float x1 = expf(x[i]);
    	float x2 = expf(-x[i]);
    	output[i] = (x1 - x2) / (x1 + x2);
    }
}

extern "C"
__global__ void tanh_backward(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] = delta[i] * (1.0f - output[i]  * output[i]);
    }
}

extern "C"
__global__ void tanh_backward_temp(float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * (1.0f - output[i]  * output[i]);
    }
}

extern "C"
__global__ void silu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

extern "C"
__global__ void silu_backward(float *x, float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] = delta[i] * (output[i] + 1.0f / (1.0f + expf(-x[i])) * (1.0f - output[i]));
    }
}

extern "C"
__global__ void silu_backward_temp(float *x, float *output, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	diff[i] += delta[i] * (output[i] + 1.0f / (1.0f + expf(-x[i])) * (1.0f - output[i]));
    }
}
