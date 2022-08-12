package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuMemAlloc;

import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class Im2colKernel {
	
	private float[] x;
	private float[] out;
	private int N;
	private int C;
	private int H;
	private int W;
	private int kh;
	private int kw;
	private int s;
	private int oHeight;
	private int oWidth;
	private int ow;
	private int oh;
	private int kSize;
	private CUfunction function;
	private int CAFFE_CUDA_NUM_THREADS = 1024;

	
	public Im2colKernel(float[] x,float[] out,int N,int C,int H,int W,int kh,int kw,int s) {
		this.x = x;
		this.N = N;
		this.C = C;
		this.H = H;
		this.W = W;
		this.kh = kh;
		this.kw = kw;
		this.s = s;
		this.oHeight = ((H - kh ) / s) + 1;
		this.oWidth = ((W - kw) / s) + 1;
		this.oh = N * oHeight * oWidth;
		this.ow = C * kh * kw;
		this.kSize = kh * kw;
		this.out = out;
		initFunction();
	}
	
	public void initFunction() {
		
		try {

			if(function == null) {
				
				function = CUDAModules.getFunctionByModule("H://Im2colKernel.cu", "im2col_gpuV6");
//				
//				/**
//				 * 加载方法
//				 */
//				CUmodule module = CUDAModules.getModule("H://Im2colKernel.cu");
//				
//		        // Obtain a function pointer to the "add" function.
//				function = new CUfunction();
//		        cuModuleGetFunction(function, module, "im2col_gpuv4");
//		        
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void im2col() {
		
		try {
//			long start1 = System.nanoTime();
			
//			long start = System.nanoTime();
			
			/**
			 * 申请内存
			 */
//			Pointer px = new Pointer();
//			JCuda.cudaMalloc(px, x.length * Sizeof.FLOAT);
//			JCuda.cudaMemcpy(px, Pointer.to(x), x.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
	        CUdeviceptr dx = new CUdeviceptr();
	        JCudaDriver.cuMemAlloc(dx, x.length * Sizeof.FLOAT);
	        JCudaDriver.cuMemcpyHtoD(dx, Pointer.to(x), x.length * Sizeof.FLOAT);
	        
	        CUdeviceptr dy = new CUdeviceptr();
	        JCudaDriver.cuMemAlloc(dy, out.length * Sizeof.FLOAT);
//	        System.out.println((System.nanoTime() - start) / 1e6 + "ms1");
//	        long start2 = System.nanoTime();
	        /**
	         * 设置入参
	         * int oHeight,int oWidth,int ow,int oh,int kSize
	         */
	        Pointer kernelParameters = Pointer.to(
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
	                Pointer.to(new int[]{oHeight * oWidth}),
	                Pointer.to(new int[]{ow}),
	                Pointer.to(new int[]{kSize})
	            );
	        
	        cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(oHeight * oWidth),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
//	        System.out.println((System.nanoTime() - start2) / 1e6 + "ms2");
//	        cuCtxSynchronize();
//	        System.out.println((System.nanoTime() - start) / 1e6 + "ms2");
	        
//	        long start3 = System.nanoTime();
//	        System.out.println(out.length);
	        JCudaDriver.cuMemcpyDtoH(Pointer.to(out), dy, out.length * Sizeof.FLOAT);
//	        System.out.println((System.nanoTime() - start3) / 1e6 + "ms3");
	        
	        // Clean up.
	        JCuda.cudaFree(dx);
	        JCuda.cudaFree(dy);
//	        System.out.println((System.nanoTime() - start1) / 1e6 + "ms1");
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

    public static void main(String args[]){	

    	int N = 128;
    	int C = 3;
    	int H = 224;
    	int W = 224;
    	int kh = 3;
    	int kw = 3;
    	int s = 1;
    	int oHeight = ((H - kh) / s) + 1;
		int oWidth = ((W - kw) / s) + 1;
		int ow = C * kh * kw;
		int oh = N * oHeight * oWidth;
    	
    	float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
    	float[][][][] x2 = MatrixUtils.transform(x, N, C, H, W);

    	float[] out = new float[oh * ow];
    	
//    	System.out.println(x.length+"start.");

		
		
//		Vector<Task<Object>> workers = new Vector<Task<Object>>();

		
		
    	for(int i = 0;i<10;i++) {
    		long start = System.nanoTime();
    		Im2colKernel k = new Im2colKernel(x, out, N, C, H, W, kh, kw, s);
        	k.im2col();
        	System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
    	}
    	
//		TaskEngine.getInstance(8).dispatchTask(workers);
		
    	
    	
//    	System.out.println(JsonUtils.toJson(out));
    	
    	System.out.println("==============================>");
		
    	float[] out2 = new float[oh * ow];

    	
    	
	    for(int i = 0;i<10;i++) {
	    	long start2 = System.nanoTime();
	//    	
	//    	float[] cpu = MatrixOperation.im2col4d(x, N, C, H, W, kh, kw, s);
	    	
	    	Im2colToVector.im2col(x2, out2, kh, kw, s);
	    	System.out.println((System.nanoTime() - start2) / 1e6 + "ms");

    	}
//    	System.out.println(JsonUtils.toJson(cpu));
	    
	    System.out.println(CheckArrayUtils.check(out, out2));
	    
    }

	public float[] getOut() {
		return out;
	}

}
