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
    	//float e = expf(-2 * x[i]);
    	//output[i] = (1 - e) / (1 + e);
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


extern "C"
__global__ void gelu_forward(float *x, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = 0.5f * x[i] * (1.0f + tanhf(0.797885 * x[i] + 0.035677f * powf(x[i], 3)));
    }
}

extern "C"
__device__ float sech_gpu(float x) {
  return 2 / (expf(x) + expf(-x)); 
}

extern "C"
__global__ void gelu_backward(float *x, float *delta, float *diff, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	const float x3 = powf(x[i], 3);
    	diff[i] = delta[i] * 0.5*tanhf(0.0356774*x3 + 0.797885*x[i]) + (0.0535161*x3 + 0.398942*x[i]) * powf(sech_gpu(0.0356774*x3 + 0.797885*x[i]), 2) + 0.5;
    }
}

__device__ float sigmoid(float x) {
    return 1.0/(1+expf(-x));
}

extern "C"
__global__ void gelu_fwd_cuda(float *x, float *output, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n) {
        output[idx] = x[idx]*sigmoid(1.702*x[idx]);
    }
}

extern "C"
__global__ void gelu_bwd_cuda(float *x, float *delta, float *diff, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n) {
        float tmp = sigmoid(1.702*x[idx]);
        diff[idx] = delta[idx]*(tmp + 1.702*x[idx]*tmp*(1-tmp));
    }
}
