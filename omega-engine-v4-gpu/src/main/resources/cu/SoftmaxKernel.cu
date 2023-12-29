#define BLOCK 1024
#define FLT_MAX 3.402823466e+38F

extern "C"
__global__ void softmax(float *input, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        output[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        output[id * n + i] /= sum;
    }
}

extern "C"
__global__ void softmaxWithTemp(float *input, float *output, int batch, int n, float temp)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i]/temp - max/temp);
        sum += e;
        output[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        output[id * n + i] /= sum;
    }
}

extern "C"
__global__ void log_softmax(float *input, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
		float t = input[id * n + i] - max;
        sum += expf(t);
        output[id * n + i] = t;
    }
	for(int i = 0;i<n;i++){
        output[id * n + i] = output[id * n + i] - log(sum);
    }
}

extern "C"
__global__ void softmax_back(float *output, float *currentLabel, float *diff, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = output[id] - currentLabel[id];    

}

extern "C"
__global__ void softmax_back2(float *output, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = (output[id] - currentLabel[id]) / batch;    

}
