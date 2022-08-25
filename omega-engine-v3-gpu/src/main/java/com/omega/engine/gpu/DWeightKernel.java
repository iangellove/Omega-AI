package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;

public class DWeightKernel {
	
	private String id;
	private float[] x;
	private float[] kernel;
	private float[] out;
	private int C;
	private int H;
	private int W;
	private int ko;
	private int kh;
	private int kw;
	private int s;
	private int p;
	private int oHeight;
	private int oWidth;
	private int ih;
	private int iw;
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUdeviceptr dx;
	private CUdeviceptr dy;
	
	private Pointer dA;
	private Pointer dC;
	
	private Pointer kernelParameters;

	public DWeightKernel(String id,float[] out,int C,int H,int W,int ko,int kh,int kw,int s,int p) {
		this.id = id;
		this.C = C;
		this.H = H;
		this.W = W;
		this.ko = ko;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.p = p;
		this.oHeight = ((H + 2 * p - kh) / s) + 1;
		this.oWidth = ((W + 2 * p - kw) / s) + 1;
		this.out = out;
		this.ih = C * kh * kw;
		this.iw = oHeight * oWidth;

		init();

	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				function = CUDAModules.getFunctionByModule("H://Im2colKernel.cu", "im2col_gpu_kernel");
        
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();

		/**
		 * 申请显存
		 */
		this.dx = CUDAMemoryManager.getDevice(C * H * W);
		this.dy = CUDAMemoryManager.getDevice(ih * iw);
		
        this.dA = CUDAMemoryManager.getPointer(ko * iw);
        this.dC = CUDAMemoryManager.getPointer(ko * ih);

        /**
         * 设置入参
         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int oh,int ow
         */
        int num_kernels = C * oHeight * oWidth; 
        
        kernelParameters = Pointer.to(
                Pointer.to(dx),
                Pointer.to(dy),
                Pointer.to(new int[]{num_kernels}),
                Pointer.to(new int[]{H}),
                Pointer.to(new int[]{W}),
                Pointer.to(new int[]{kh}),
                Pointer.to(new int[]{kw}),
                Pointer.to(new int[]{s}),
                Pointer.to(new int[]{oHeight}),
                Pointer.to(new int[]{oWidth})
            );
        
	}
	
	public void setX(float[] x) {
		this.x = x;
		/**
		 * k * n
		 */
        JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT);
	}
	
	public void setKernel(float[] kernel) {
		this.kernel = kernel;
		/**
		 * m * n
		 */
		cublasSetVector(ko * iw, Sizeof.FLOAT, Pointer.to(kernel), 1, dA, 1);
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void conv() {

		im2col();
		
		sgemm();

	}
	
	public void sgemm() {
		/**
		 * m k n
		 */
		GPUOP.getInstance().multiplyFloat(ko, ih, iw, dA, dy, dC, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 1.0f);
		
	}
	
	public void im2col() {
		
		try {

	        cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(ih),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
	        
	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void free() {
		 JCuda.cudaFree(dx);
	     JCuda.cudaFree(dy);
	     if(dA!=null) {
	    	 GPUOP.getInstance().free(dA);
		     GPUOP.getInstance().free(dC);
	     }
	}
	
	public float[] getOut() {
		cublasGetVector(out.length, Sizeof.FLOAT, dC, 1, Pointer.to(out), 1);
		return out;
	}
	
	public void clear() {
		out = new float[out.length];
		cublasSetVector(out.length, Sizeof.FLOAT, Pointer.to(out), 1, dC, 1);
	}

    public static void main(String args[]){	
    	int N = 2;
    	int C = 3;
    	int H = 12;
    	int W = 12;
    	int ko = 20;
    	int kh = 5;
    	int kw = 5;
    	int s = 1;
    	int p = 0;
    	int oHeight = ((H + 2 *  - kh) / s) + 1;
		int oWidth = ((W - kw) / s) + 1;
		int ow = C * kh * kw;
		int oh = ko;
		
    	float[] x1 = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
    	
    	float[] k1 = RandomUtils.gaussianRandom(N * ko * oHeight * oWidth, 0.1f);
    	
    	float[] out = new float[oh * ow];

    	float[] once = new float[C * H * W];
    	
    	float[] kOnce = new float[ko * oHeight * oHeight];
    	
    	DWeightKernel ck = new DWeightKernel("conv1", out, C, H, W, ko, kh, kw, s, p);

    	long start = System.nanoTime();
    	
    	for(int n = 0;n<N;n++) {
//    		long start2 = System.nanoTime();
    		System.arraycopy(x1, n * C * H * W, once, 0, C * H * W);
    		System.arraycopy(k1, n * ko * oHeight * oHeight, kOnce, 0, ko * oHeight * oHeight);
    		ck.setX(once);
    		ck.setKernel(kOnce);
        	ck.conv();
        	System.out.println(JsonUtils.toJson(ck.getOut()));
//        	System.arraycopy(ck.getOut(), 0, out2, i * oh * ow, oh * ow);
//        	MatrixUtils.col2im4d(ck.getOut(), out2, n, ko, oHeight, oWidth);
//        	System.out.println((System.nanoTime() - start2) / 1e6 + "ms.:"+i);
    	}


		CUDAMemoryManager.free();
    }
	
}
