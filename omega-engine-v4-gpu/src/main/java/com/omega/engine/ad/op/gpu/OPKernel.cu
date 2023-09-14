#define BLOCK 1024 

extern "C"
__global__ void fill_kernel(int N, float ALPHA, float *X)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i] = ALPHA;
}

extern "C"
__global__ void copy_kernel(int N,  float *X, int OFFX, float *Y, int OFFY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
    	Y[i + OFFY] = X[i + OFFX];
    }
}

extern "C"
__global__ void copy_number_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start,int cp)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
    	int size = c * h * w;
    	int tn = i / size + start;
		int tc = (i / h / w) % c;
		int th = (i / w) % h;
		int tw = i % h;
		int index = tn * size + tc * h * w + th * w + tw;
		if(cp == 0){
			Y[i] = X[index];
		}else{
			X[index] = Y[i];
		}
    }
}

extern "C"
__global__ void copy_channel_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start,int cp)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
    	int bc = N / n / h / w;
		int size = bc * h * w;
    	int tn = i / size;
		int tc = (i / h / w) % bc + start;
		int th = (i / w) % h;
		int tw = i % h;
		int index = tn * c * h * w + tc * h * w + th * w + tw;
    	if(cp == 0){
			Y[i] = X[index];
		}else{
			X[index] = Y[i];
		}
    }
}

extern "C"
__global__ void broadcast_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] = X[0];
}

extern "C"
__global__ void broadcast_number_kernel(int N, float *X, float *Y,int C,int H,int W)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
    	int n = i / C / H / W;
    	Y[i] = X[n];
    }
}

extern "C"
__global__ void add_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] + Y[i];
}

extern "C"
__global__ void sum_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < 1) {
	    for(int index = 0;index<N;index++){
	    	Y[0] += X[index];
	    }
    }
}

extern "C"
__global__ void sum_channel_kernel(int N, float *X, float *Y,int C,int H,int W)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
    	for(int index = 0;index<C * H * W;index++){
    		Y[i] += X[i * C * H * W + index];
    	}
    }
}

extern "C"
__global__ void add_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] + ALPHA;
}

extern "C"
__global__ void add_number_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
    	int size = c * h * w;
    	int tn = i / size + start;
		int tc = (i / h / w) % c;
		int th = (i / w) % h;
		int tw = i % h;
		int index = tn * size + tc * h * w + th * w + tw;
    	X[index] += Y[i];
    }
}

extern "C"
__global__ void add_channel_kernel(int N,  float *X, float *Y, int n,int c,int h,int w,int start)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N){
    	int bc = N / n / h / w;
		int size = bc * h * w;
    	int tn = i / size;
		int tc = (i / h / w) % bc + start;
		int th = (i / w) % h;
		int tw = i % h;
		int index = tn * c * h * w + tc * h * w + th * w + tw;
    	X[index] += Y[i];
    }
}

extern "C"
__global__ void sub_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] - Y[i];
}

extern "C"
__global__ void sub_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] - ALPHA;
}

extern "C"
__global__ void scalar_sub_kernel(int N, float ALPHA, float *X, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = ALPHA - X[i];
}

extern "C"
__global__ void mul_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] * Y[i];
}

extern "C"
__global__ void mul_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] * ALPHA;
}

extern "C"
__global__ void mul_plus_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] += X[i] * Y[i];
}

extern "C"
__global__ void mul_plus_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] += X[i] * ALPHA;
}

extern "C"
__global__ void div_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] / Y[i];
}

extern "C"
__global__ void div_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = X[i] / ALPHA;
}

extern "C"
__global__ void scalar_div_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] = ALPHA / X[i];
}

extern "C"
__global__ void div_bGrad_kernel(int N, float *A, float *B, float *C, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] += - 1.0f * C[i] * A[i] / (B[i] * B[i]); 
}

extern "C"
__global__ void div_scalar_bGrad_kernel(int N, float *D, float A, float *B, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] += - 1.0f * D[i] * A / (B[i] * B[i]); 
}

extern "C"
__global__ void div_plus_kernel(int N, float *X, float *Y, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] += X[i] / Y[i];
}

extern "C"
__global__ void div_plus_scalar_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] += X[i] / ALPHA;
}

extern "C"
__global__ void scalar_plus_div_kernel(int N, float *X, float ALPHA, float *R)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) R[i] += ALPHA / X[i];
}

extern "C"
__global__ void pow_kernel(int N, float *X, float ALPHA, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] = powf(X[i], ALPHA);
}

extern "C"
__global__ void log_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
    	//if(X[i] == 0){
    		//X[i] = 0.00000000000000000000001f;
    	//}
   	 	Y[i] = logf(X[i]);
    }
}

extern "C"
__global__ void exp_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] = expf(X[i]);
}

extern "C"
__global__ void sin_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] = sin(X[i]);
}

extern "C"
__global__ void cos_kernel(int N, float *X, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i] = cos(X[i]);
}

extern "C"
__global__ void clamp_kernel(int N, float *X, float min, float max, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
    	float val = X[i];
		if(val < min) {
			Y[i] = min;
		}else if(val > max) {
			Y[i] = max;
		}else {
			Y[i] = val;
		}
    }
}

extern "C"
__global__ void clamp_back_kernel(int N, float *X, float min, float max, float *Y)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
    	float val = X[i];
		if(val < min || val > max) {
			Y[i] = 0;
		}else {
			Y[i] = 1;
		}
    }
}
