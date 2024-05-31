package com.omega.engine.ad.op.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.io.Serializable;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.gpu.LNKernel;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

public class NormalizeKernel implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3345793649705471080L;

	public int N = 0;
	
	private static NormalizeKernel kernel = null;

	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUfunction norm_function;
	
//	private CUfunction norm_grad_function;
	
	public NormalizeKernel() {
		
		norm_function = CUDAModules.getLocalFunctionByModule("NormalizeKernel.cu", "norm");
		
//		norm_grad_function = CUDAModules.getLocalFunctionByModule("NormalizeKernel.cu", "NormalizeGradientKernel"); 
		
	}
	
	public static NormalizeKernel getInstance() {
		if(kernel == null) {
			kernel = new NormalizeKernel();
		}
		return kernel;
	}
	
	public void norm(Tensor x,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{x.getDataLength()}),
	                Pointer.to(x.getGpuData()),
	                Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(norm_function,
	        		CAFFE_GET_BLOCKS(x.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public static void main(String[] args) {
    	
    	int N = 1024;
    	int W = 1024;
    	
    	float[] data = RandomUtils.order(N * W, 0.1f, 0.1f);
    	
    	Tensor input = new Tensor(N, 1, 1, W, data, true);
    	
    	Tensor output = new Tensor(1, 1, 1, 1, true);
    	
    	NormalizeKernel kernel = new NormalizeKernel();
    	
    	Tensor input2 = new Tensor(N, 1, 1, W, data, true);
    	
    	Tensor output2 = new Tensor(1, 1, 1, 1, true);

    	
//    	long start = System.nanoTime();
//    	for(int i = 0;i<10;i++) {
////        	System.out.println("output:");
//        	kernel.norm(input, output);
////        	output.showDM();
//    	}
//    	output.showDM();
//    	System.out.println((System.nanoTime() - start)/1e6+"ms.");
    	
    	long start2 = System.nanoTime();
    	for(int i = 0;i<10;i++) {
    		output2.valueGPU(0);
    		TensorOP.pow(input, 2, input2);
        	TensorOP.sum(input2, output2, 0);
        	TensorOP.sqrt(output2, output2);
    	}
    	output2.showDM();
    	System.out.println((System.nanoTime() - start2)/1e6+"ms.");
    	
	}
}
