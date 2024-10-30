#define BLOCK 1024 

#define _USE_MATH_DEFINES
#include <math.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

extern "C"
__global__ void reparameterize_forward(float *mu,float *logvar,float *eps, float *output, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	output[i] = eps[i] * expf(0.5f * logvar[i]) + mu[i];
    }
}

extern "C"
__global__ void reparameterize_backward(float *dmu,float *dlogvar,float *eps,float *logvar, float *delta, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	dlogvar[i] += delta[i] * eps[i] * expf(0.5f * logvar[i]) * 0.5f;
    	dmu[i] += delta[i];
    }
}


extern "C"
__global__ void kl_loss(float *mu,float *logvar,float kl_weight, float *klLoss, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	klLoss[i] = -0.5f * (1 + logvar[i] - powf(mu[i], 2) - expf(logvar[i])) * kl_weight;
    }
}

extern "C"
__global__ void kl_loss_back(float *mu,float *logvar,float kl_weight, float *dmu, float * dlogvar,int batch, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) {
    	dmu[i] = kl_weight * mu[i];
    	dlogvar[i] = kl_weight * -0.5f * (1.0f - expf(logvar[i]));
    }
}