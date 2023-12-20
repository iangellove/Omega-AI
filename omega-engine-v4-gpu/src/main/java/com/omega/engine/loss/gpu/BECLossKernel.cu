#define BLOCK 1024

extern "C"
__global__ void loss(float *input, float *label, float *output, int batch, int n,float eta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch) return;

	float sum = 0;
	float neg_100 = -100;
	float one = 1;
	float EPSILON = 1e-12f;

	for(int i = 0;i<n;i++){
        //sum += (label[i] * logf(input[i] + eta) + (1 - label[i]) * logf((1 - input[i]) + eta));
        float input_val = fmax(input[i], EPSILON);
        float log_input_val = logf(input_val);
        float log_1_minus_input_val = logf(1 - input_val);
        log_input_val = fmax(log_input_val, neg_100);
        log_1_minus_input_val = fmax(log_1_minus_input_val, neg_100);
        sum += ((label[i] - one) * log_1_minus_input_val) - (label[i] * log_input_val);
    }
    
	output[id] = sum / n;
}

extern "C"
__global__ void loss_back(float *input, float *currentLabel, float *diff, int n, int batch)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= n) return;
	
	//diff[id] = - (currentLabel[id] / input[id] + (1 - currentLabel[id]) / (1 - input[id])) / batch / n;    
	
	float one = 1;
	float EPSILON = 1e-12f;
	
	float grad_input_denominator = fmax((one - input[id]) * input[id], EPSILON);
	
	diff[id] = (input[id] - currentLabel[id]) / grad_input_denominator / batch / n;
}
