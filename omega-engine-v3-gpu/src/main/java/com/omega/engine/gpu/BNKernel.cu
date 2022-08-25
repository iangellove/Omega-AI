#define BLOCK 1024 


extern "C"
__global__ void culOutput_cov(float* x,float* mean,float* std,float* gama,float* beta,float* z,float* out,int N,int number,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < N; index += blockDim.x * gridDim.x) {
		
		int c = index / (height * width);
		
		//int h = index / width;
		
		//int w = index % width;
	
		float mean_val = mean[c];
		float std_val = std[c];
		float gama_val = gama[c];
		float beta_val = beta[c];
	
		for(int n = 0;n<number;n++) {	
			
			float z_val = (x[n * N + index] - mean_val) / std_val;
			
			z[n * N + index] = z_val;
			
			out[n * N + index] = z_val * gama_val + beta_val;
			
		}	
		
	}

}

extern "C"
__global__ void culOutput_full(float* x,float* mean,float* std,float* gama,float* beta,float* z,float* out,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < width; index += blockDim.x * gridDim.x) {
		
		float mean_val = mean[index];
		float std_val = std[index];
		float gama_val = gama[index];
		float beta_val = beta[index];
	
		for(int n = 0;n<number;n++) {	
			
			float z_val = (x[n * width + index] - mean_val) / std_val;
			
			z[n * width + index] = z_val;
			
			out[n * width + index] = z_val * gama_val + beta_val;
			
		}	
		
	}

}


extern "C"
__global__ void computeDelta(float* delta,float* gama,float* deltaGama,float* deltaBeta,float* z,float* diff,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		float gama_val = gama[index];
		
		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					float delta_val = delta[n * channel * height * width + index * height * width + h * width + w];
					
					deltaGama_val += delta_val * z[n * channel * height * width + index * height * width + h * width + w];
					
					deltaBeta_val += delta_val;
					
					diff[n * channel * height * width + index * height * width + h * width + w] = delta_val * gama_val;
	
				}
			}
		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}

extern "C"
__global__ void meanDzSum(float* x,float* dz,float* mean,float* var,float* std,float* dvar,float* dmu,float scale,int number,int channel,int height,int width,float eta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
		
		float std_val = std[index];
		float mean_val = mean[index];
		
		float dvar_val = 0.0f;
		float dmu_val = 0.0f;
		float dmu2_val = 0.0f;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					float x_val = x[n * channel * height * width + index * height * width + h * width + w];
					float dz_val = dz[n * channel * height * width + index * height * width + h * width + w];
					
					dvar_val += (x_val - mean_val) * dz_val;
					dmu_val += -1.0f * dz_val / std_val;
					dmu2_val += -2.0f * (x_val - mean_val) * scale;
				}
			}
		}	
		
		dvar[index] = dvar_val * -0.5 * pow(var[index] + eta, -1.5);
		dmu[index] = dmu_val + dmu2_val * dvar[index];
		
	}

}


extern "C"
__global__ void computeDiff(float* x,float* dvar,float* dmu,float* mean,float* std,float* diff,float scale,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
		
		float dvar_val = dvar[index];
		float dmu_val = dmu[index];
		float std_val = std[index];
		float mean_val = mean[index];
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					diff[xIndex] = diff[xIndex] / std_val + 2.0 * dvar_val * (x[xIndex] - mean_val) * scale + dmu_val * scale;
				}
			}
		}	
		
	}

}


extern "C"
__global__ void computeDgama(float* delta,float* deltaGama,float* deltaBeta,float* z,float* diff,float* gama,int number,int channel,int height,int width)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {
	
		float gama_val = gama[index];
	
		float deltaGama_val = 0;
		
		float deltaBeta_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					float delta_val = delta[n * channel * height * width + index * height * width + h * width + w];
					
					deltaGama_val += delta_val * z[n * channel * height * width + index * height * width + h * width + w];
					
					deltaBeta_val += delta_val;
					
					diff[n * channel * height * width + index * height * width + h * width + w] = delta_val * gama_val;
					
				}
			}
		}	
		
		deltaGama[index] = deltaGama_val;
		deltaBeta[index] = deltaBeta_val;
		
	}

}


extern "C"
__global__ void meanDelta(float* delta,float* out,float* meanDelta,float* meanDeltaDot,int number,int channel,int height,int width,float scale)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		float meanDelta_val = 0;
		
		float meanDeltaDot_val = 0;
		
		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
					
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					
					meanDelta_val += delta[xIndex] * scale;
					
					meanDeltaDot_val += delta[xIndex] * out[xIndex] * scale;
					
				}
			}
		}	
		
		meanDelta[index] = meanDelta_val;
		meanDeltaDot[index] = meanDeltaDot_val;
		
	}

}


extern "C"
__global__ void dx_fn(float* delta,float* out,float* diff,float* meanDelta,float* meanDeltaDot,float* std,int number,int channel,int height,int width,float eta)
{
    
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < channel; index += blockDim.x * gridDim.x) {

		for(int n = 0;n<number;n++) {	
			for(int h = 0;h<height;h++) {
				for(int w = 0;w<width;w++) {
				
					int xIndex = n * channel * height * width + index * height * width + h * width + w;
					
					diff[xIndex] = (delta[xIndex] - meanDelta[index] - meanDeltaDot[index] * out[xIndex]) / sqrt((std[index] + eta));
					
				}
			}
		}	
		
	}

}


extern "C"
__global__ void normalize_kernel(int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    float val = (x[index] - mean[f])/(sqrtf(variance[f] + 0.00001f));
    z[index] = val;
    out[index] = val * gama[f] + beta[f];
}


extern "C"
__global__ void dgama_kernel(float *xhat, float *delta, int batch ,int c, int size, float *dgama)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + c*b);
            sum += (p+i < size) ? delta[index]*xhat[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) dgama[filter] += part[i];
    }
}


extern "C"
__global__ void dbeta_kernel(float *dbeta, float *delta, int batch, int c, int size)
{
    __shared__ float part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + c*b);
            sum += (p+i < size) ? delta[index] : 0;
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) dbeta[filter] += part[i];
    }
}


extern "C"
__global__ void dxhat_kernel2(int N, float *delta, float *dz, float *gama, int filters, int spatial)
{	
	int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;

    dz[index] = delta[index] * gama[f];
}


extern "C"
__global__ void dxhat_kernel(float *dz, float *gama, int c, int size)
{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) dz[(batch*c+filter)*size + offset] *= gama[filter];
}

//dmean = (∑ dxhat * -1 / (var + eta)^1/2)
extern "C"
__global__ void fast_mean_delta_kernel(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (-1.f/sqrtf(variance[filter] + .00001f));
    }
}

//dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
extern "C"
__global__ void  fast_variance_delta_kernel(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{
    const int threads = BLOCK;
    __shared__ float local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= -.5f * powf(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}


//dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/2
extern "C"
__global__ void dx_kernel(int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *diff)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    diff[index] = diff[index] * 1.f/(sqrtf(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial * batch);
}

