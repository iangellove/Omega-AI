#define BLOCK 1024

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n,float eta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;

	for(int i = 0;i<n;i++){
        sum += (label[i] * logf(input[i] + eta) + (1 - label[i]) * logf((1 - input[i]) + eta));
    }
    
	output[id] = sum / n;
}

extern "C"
__global__ void loss_back(float *output, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	diff[id] = - (currentLabel[id] / output[id] + (1 - currentLabel[id]) / (1 - output[id])) / batch / n;    

}
