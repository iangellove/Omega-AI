#define BLOCK 1024
#define FLT_MAX 3.402823466e+38F

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += - label[id * n + i] * log(input[id * n + i]);
    }
    
	output[id] = sum;
}

extern "C"
__global__ void nl_loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += - label[id * n + i] * input[id * n + i];
    }
    
	output[id] = sum;
}

extern "C"
__global__ void log_softmax_nl_loss(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float loss_sum = 0;
	float EPSILON = 1e-12f;
	
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        sum += expf(input[id * n + i] - max);
    }
	for(int i = 0;i<n;i++){
        //loss_sum += - ((input[id * n + i] - max) - logf(sum)) * label[id * n + i];
        float vl = fmax(expf(input[id * n + i] - max) / sum, EPSILON);
        loss_sum -= logf(vl) * label[id * n + i];
    }
    output[id] = loss_sum;
}

extern "C"
__global__ void log_softmax_nl_loss_igone(float *input, float *label, float *output, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float loss_sum = 0;
	float EPSILON = 1e-12f;
	if(label[id * n + igonre] == 1){
        output[id] = 0.0f;
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        sum += expf(input[id * n + i] - max);
    }
	for(int i = 0;i<n;i++){
        //loss_sum += - ((input[id * n + i] - max) - logf(sum)) * label[id * n + i];
        float vl = fmax(expf(input[id * n + i] - max) / sum, EPSILON);
        loss_sum -= logf(vl) * label[id * n + i];
    }
    output[id] = loss_sum;
}

extern "C"
__global__ void check(float *input, float *label, float *output, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	for(int i = 0;i<n;i++){
        output[id * n + i] =  - label[id * n + i] * log(input[id * n + i]);
    }

}

extern "C"
__global__ void loss_back(float *output, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = (output[id] - currentLabel[id]) / batch;    

}

extern "C"
__global__ void loss_back2(float *input, float *currentLabel, float *diff, int batch, int n)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float EPSILON = 1e-12f;
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        diff[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        //diff[id * n + i] = ((diff[id * n + i] / sum) - currentLabel[id * n + i]) / batch;
        float vl = fmax(diff[id * n + i] / sum, EPSILON);
        diff[id * n + i] = vl - currentLabel[id * n + i];
    } 
}

extern "C"
__global__ void loss_back_igonre2(float *input, float *currentLabel, float *diff, int batch, int n,int igonre)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;
	float max = -FLT_MAX;
	float sum = 0;
	float EPSILON = 1e-12f;
	if(currentLabel[id * n + igonre] == 1){
		for(int i = 0;i<n;i++){
        	diff[id * n + i] = 0.0f;
        }
        return;
    }
	for(int i = 0;i<n;i++) {
		if(max <= input[id * n + i]) {
			max = input[id * n + i];
		}
	}
	for(int i = 0;i<n;i++){
        float e = expf(input[id * n + i] - max);
        sum += e;
        diff[id * n + i] = e;
    }
	for(int i = 0;i<n;i++){
        //diff[id * n + i] = ((diff[id * n + i] / sum) - currentLabel[id * n + i]) / batch;
        float vl = fmax(diff[id * n + i] / sum, EPSILON);
        diff[id * n + i] = vl - currentLabel[id * n + i];
    }
}
