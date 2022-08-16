extern "C"
__global__ void im2col_gpu(float[][][][] x,float[][][][][] mask,int pWidth,int pHeight,int stride,PoolingType poolingType,float[][][][] result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < oh){
   		
		int n = i / oHeight / oWidth;
				
		int startH = (i - (n * oHeight * oWidth)) / oWidth * stride;
		
		int startW = (i - (n * oHeight * oWidth)) % oWidth * stride;
	
		for(int j = 0;j<ow;j++) {
			
			int c = j / kSize;
				
			int xSize = j - (c * kSize);
			
			int xh = startH + xSize / kh;
			
			int xw = startW + xSize % kw;
			
			out[i * ow + j] = x[n * C * H * W + c * H * W + xh * W + xw];
	
		}
   		
    }
   
}
