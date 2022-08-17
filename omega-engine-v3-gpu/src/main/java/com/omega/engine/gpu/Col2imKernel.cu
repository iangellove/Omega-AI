extern "C"
__global__ void col2im_gpu_kernel(float* data_col,float* data_im,int n,const int height, const int width, const int channels, const int ksize,int pad, int stride, int height_col, int width_col) {
  	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < n; index += blockDim.x * gridDim.x) {
        float val = 0;
        int w = index % width + pad;
	     int h = (index / width) % height + pad;
	     int c = index / (width * height);
	     // compute the start and end of the output
	     int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
	     int w_col_end = min(w / stride + 1, width_col);
	     int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
	     int h_col_end = min(h / stride + 1, height_col);
	 
	     // equivalent implementation
	     int offset = (c * ksize * ksize + h * ksize + w) * height_col * width_col;
	     int coeff_h_col = (1 - stride * ksize * height_col) * width_col;
	     int coeff_w_col = (1 - stride * height_col * width_col);
	     for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
	       for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
	         val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
	       }
	     }
        data_im[index] = val;
  	}
}
