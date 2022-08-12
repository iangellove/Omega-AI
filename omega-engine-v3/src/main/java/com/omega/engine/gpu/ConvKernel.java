package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.runtime.JCuda.cudaMalloc;

import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class ConvKernel {
	
	private float[] x;
	private float[] kernel;
//	private float[] x_im2col;
	private float[] out;
	private int N;
	private int C;
	private int H;
	private int W;
	private int ko;
	private int kh;
	private int kw;
	private int s;
	private int oHeight;
	private int oWidth;
	private int oh;
	private int ow;
	
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUdeviceptr dx;
	private CUdeviceptr dy;
	
	private Pointer dA;
	private Pointer dB;
	private Pointer dC;
	
	private Pointer kernelParameters;

	public ConvKernel(float[] x,float[] kernel,float[] out,int N,int C,int H,int W,int ko,int kh,int kw,int s) {
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.ko = ko;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.oHeight = ((H - kh ) / s) + 1;
		this.oWidth = ((W - kw) / s) + 1;
		this.oh = N * oHeight * oWidth;
		this.ow = C * kh * kw;
		this.x = x;
		this.out = out;
		this.kernel = kernel;
		/**
		 * 初始化cuda函数
		 */
		initFunction();

		initIm2col();
		
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				function = CUDAModules.getFunctionByModule("H://Im2colKernel.cu", "im2col_gpuV6");
        
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void initIm2col() {
		
		/**
		 * 申请显存
		 */
		this.dx = new CUdeviceptr();
		checkCUDA(JCudaDriver.cuMemAlloc(dx, x.length * Sizeof.FLOAT));
		
		this.dy = new CUdeviceptr();
		checkCUDA(JCudaDriver.cuMemAlloc(dy, oh * ow * Sizeof.FLOAT));
		
	}
	
	public void initConv() {

		/**
		 * 申请显存
		 */
		this.dA = dy;
		this.dB = new Pointer();
		this.dC = new Pointer();

		checkCUDA(cudaMalloc(dB, ow * ko * Sizeof.FLOAT));
		checkCUDA(cudaMalloc(dC, oh * ko * Sizeof.FLOAT));
		
		checkCUDA(cublasSetVector(ow * ko, Sizeof.FLOAT, Pointer.to(kernel), 1, dB, 1));
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void conv() {

//		long start = System.nanoTime();

		try {

			im2col();
			
			sgemm();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {

			free();
			
		}
		
//		System.out.println((System.nanoTime() - start) / 1e6 + "ms22222");
        
	}
	
	public void sgemm() {
		
		initConv();
		
		GPUOP.getInstance().multiplyFloat(oh, ow, ko, this.out, dA, dB, dC);
		
	}
	
	public void im2col() {
		
		try {
			

			/**
			 * 申请内存
			 */
			checkCUDA(JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT));
			
//	        long start3 = System.nanoTime();

	        /**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int oh,int ow
	         */
	        int num_kernels = oHeight * oWidth; 
	       
	        kernelParameters = Pointer.to(
	        		Pointer.to(dx),
	                Pointer.to(dy),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{C}),
	                Pointer.to(new int[]{H}),
	                Pointer.to(new int[]{W}),
	                Pointer.to(new int[]{kh}),
	                Pointer.to(new int[]{kw}),
	                Pointer.to(new int[]{s}),
	                Pointer.to(new int[]{oHeight}),
	                Pointer.to(new int[]{oWidth}),
	                Pointer.to(new int[]{num_kernels}),
	                Pointer.to(new int[]{ow}),
	                Pointer.to(new int[]{kh * kw})
	            );
	        long start1 = System.nanoTime();
	        checkCUDA(cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(num_kernels),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));
	       
	        checkCUDA(JCudaDriver.cuCtxSynchronize());
	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms22222");

	        // Clean up.
	        JCuda.cudaFree(dx);
//	        JCuda.cudaFree(dy);
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void free() {
//		if(dx != null) {
//			JCuda.cudaFree(dx);
//		}
		if(dy != null) {
			JCuda.cudaFree(dy);
		}
		if(dA != null) {
			GPUOP.getInstance().free(dA);
		}
		if(dB != null) {
			GPUOP.getInstance().free(dB);
		}
		if(dC != null) {
			GPUOP.getInstance().free(dC);
		}
	}
	
    public static void main(String args[]){	
    	int N = 1;
    	int C = 1;
    	int H = 5;
    	int W = 5;
    	int ko = 1;
    	int kh = 3;
    	int kw = 3;
    	int s = 1;
    	int oHeight = ((H - kh) / s) + 1;
		int oWidth = ((W - kw) / s) + 1;
		int ow = N * oHeight * oWidth;
		int oh = ko;

    	float[] x1 = MatrixUtils.order(N * C * H * W, 1);
    	
    	float[] k1 = MatrixUtils.order(ko * C * kh * kw, 1);

    	float[] out = new float[oh * ow];

    	float[] out1 = new float[oh * ow];
    	
    	int ow2 = C * kh * kw;
		int oh2 = N * oHeight * oWidth;
    	
    	float[] im2col = new float[oh2 * ow2];
    	
    	float[][][][] x2 = MatrixUtils.transform(x1, N, C, H, W);
    	
    	float[][][][] k2 = MatrixUtils.transform(k1, ko, C, kh, kw);

    	float[] ka = Im2colUtils.kernalToVector(k2, false);
		
    	long start2 = System.nanoTime();
    	
    	for(int i = 0;i<20;i++) {
    		long start1 = System.nanoTime();

        	Im2colToVector.im2col(x2, im2col, kh, kw, s);

        	float[] r = new float[oh * ow];
        	
        	int xm = N * oHeight * oWidth;
    		int xn = kh * kw * C;
        	
    		GPUOP.getInstance().multiplyFloat(xm, xn, ko, im2col, ka, r);

	    	System.out.println((System.nanoTime() - start1) / 1e6 + "ms.cpu:"+i);
	    	out1 = r;
    	}
    	
		System.out.println( "CPU=================>:"+(System.nanoTime() - start2) / 1e6+"ms.");
		

    	long start = System.nanoTime();
    	
		for(int c = 0;c<20;c++){

	    	long start3 = System.nanoTime();

			ConvKernel ck = new ConvKernel(x1, ka, out, N, C, H, W, ko, kh, kw, s);

	    	ck.conv();
	    	
	    	System.out.println((System.nanoTime() - start3) / 1e6 + "ms.gpu:"+c);
	    	out = ck.getOut();
		}
		
		System.out.println( "GPU=================>:"+(System.nanoTime() - start) / 1e6+"ms.");
		
		System.out.println("error:"+CheckArrayUtils.check(out, out1));
		
    }

	public float[] getOut() {
		return out;
	}

}
