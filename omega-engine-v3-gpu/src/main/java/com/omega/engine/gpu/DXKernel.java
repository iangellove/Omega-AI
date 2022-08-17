package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class DXKernel {
	
	private String id;
	private float[] kernel;
	private float[] delta;
	private float[] out;
	private int C;
	private int H;
	private int W;
	private int ko;
	private int kh;
	private int kw;
	private int pad;
	private int s;
	private int oHeight;
	private int oWidth;
	private int ih;
	private int iw;
	private int numKernels;
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUdeviceptr dy;
	
	private Pointer dA;
	private Pointer dB;
	private Pointer dC;
	
	private Pointer kernelParameters;

	public DXKernel(String id,float[] out,int C,int H,int W,int ko,int kh,int kw,int pad,int s) {
		this.id = id;
		this.C = C;
		this.H = H;
		this.W = W;
		this.ko = ko;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.pad = pad;
		this.oHeight = ((H + 2 * pad - kh ) / s) + 1;
		this.oWidth = ((W + 2 * pad - kw) / s) + 1;
		this.out = out;
		this.ih = C * kh * kw;
		this.iw = oHeight * oWidth;
		this.numKernels = C * H * W;
		init();
		
//        System.out.println((System.nanoTime() - start1) / 1e6 + "ms.1111");
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				function = CUDAModules.getFunctionByModule("H://Col2imKernel.cu", "col2im_gpu_kernel");
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
		
//		/**
//		 * 申请内存
//		 */
//		if(x_im2col == null) {
//			x_im2col = new float[ih * iw];
//		}

		/**
		 * 申请显存
		 */
		this.dy = CUDAMemoryManager.getDevice(id + "_dxy", C * H * W);
		
        this.dA = CUDAMemoryManager.getPointer(id + "_dxA", ko * ih);
        this.dB = CUDAMemoryManager.getPointer(id + "_dxB", ko * iw);
        this.dC = CUDAMemoryManager.getPointer(id + "_dxC", ih * iw);
		
        /**
         * 设置入参
         * float* data_col,float* data_im,int n,int height,int width,int channels,int ksize,int pad,int stride,int height_col,int width_col
         */
        kernelParameters = Pointer.to(
        		Pointer.to(dC),
                Pointer.to(dy),
                Pointer.to(new int[]{numKernels}),
                Pointer.to(new int[]{H}),
                Pointer.to(new int[]{W}),
                Pointer.to(new int[]{C}),
                Pointer.to(new int[]{kh}),
                Pointer.to(new int[]{pad}),
                Pointer.to(new int[]{s}),
                Pointer.to(new int[]{oHeight}),
                Pointer.to(new int[]{oWidth})
            );
        
	}
	
	public void setKernel(float[] kernel) {
		this.kernel = kernel;
		cublasSetVector(ko * ih, Sizeof.FLOAT, Pointer.to(kernel), 1, dA, 1);
	}

	public void setDelta(float[] delta) {
		this.delta = delta;
		cublasSetVector(ko * iw, Sizeof.FLOAT, Pointer.to(delta), 1, dB, 1);
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void conv() {

		sgemm();

		col2im();
		
	}
	
	public void sgemm() {
		
		GPUOP.getInstance().multiplyFloat(ih, iw, ko, dA, dB, dC, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 0.0f, 1.0f);
		
	}
	
	public void col2im() {
		
		try {

			checkCUDA(cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(numKernels),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public float[] getOut() {
		JCudaDriver.cuMemcpyDtoH(Pointer.to(out), dy, out.length * Sizeof.FLOAT);
		return out;
	}

	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
    public static void main(String args[]){	
    	int N = 3;
    	int C = 20;
    	int H = 12;
    	int W = 12;
    	int ko = 50;
    	int kh = 5;
    	int kw = 5;
    	int s = 1;
    	int p = 0;
    	int oHeight = ((H + 2 * p - kh) / s) + 1;
		int oWidth = ((W + 2 * p - kw) / s) + 1;
		
    	float[] x1 = RandomUtils.gaussianRandom(N * ko * oHeight * oWidth, 0.1f);
    	
    	float[] k1 = RandomUtils.gaussianRandom(ko * C * kh * kw, 0.1f);
    	
    	float[] out = new float[C * H * W];

    	float[] once = new float[ko * oHeight * oWidth];
    	
    	DXKernel ck = new DXKernel("conv1", out, C, H, W, ko, kh, kw, p, s);
    	
    	ck.setKernel(k1);
    	
//    	long start = System.nanoTime();
    	
    	for(int n = 0;n<N;n++) {
//    		long start2 = System.nanoTime();
    		System.arraycopy(x1, n * ko * oHeight * oWidth, once, 0, ko * oHeight * oWidth);
    		ck.setDelta(once);
        	ck.conv();
        	System.out.println(JsonUtils.toJson(ck.getOut()));
//        	System.arraycopy(ck.getOut(), 0, out2, i * oh * ow, oh * ow);
//        	MatrixUtils.col2im4d(ck.getOut(), out2, n, ko, oHeight, oWidth);
//        	System.out.println((System.nanoTime() - start2) / 1e6 + "ms.:"+i);
    	}


		CUDAMemoryManager.free();
    }
	
}
