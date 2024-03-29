package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class DropoutKernel extends BaseKernel{

	private CUfunction function;
	
	private CUfunction back_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer kernelParameters;
	
	private Pointer kernelBackParameters;
	
	private float prob = 0.0f;
	
	private float scale = 1.0f;
	
	public DropoutKernel(float prob, float scale) {
		this.prob = prob;
		this.scale = scale;
		init();
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();

	}
	
	public void initFunction() {
		
		try {

			if(function == null) {

				function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"dropout.cu", "forward_kernel");
				
			}
			
			if(back_function == null) {

				back_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"dropout.cu", "backward_kernel");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor input,Tensor rand) {
		
		try {
			
			if(kernelParameters == null || input.number != this.N){

		        /**
		         * 设置入参
		         * float *input, int size, float *rand, float prob, float scale
		         */ 
		        kernelParameters = Pointer.to(
		        		Pointer.to(input.getGpuData()),
		                Pointer.to(new int[]{input.getDataLength()}),
		                Pointer.to(rand.getGpuData()),
		                Pointer.to(new float[]{prob}),
		                Pointer.to(new float[]{scale})
		            );
		        
		        this.N = input.number;
		        
			}

			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(input.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor delta,Tensor rand) {
		
		try {
			
			if(kernelBackParameters == null || delta.number != this.N){

		        /**
		         * 设置入参
		         * float *input, int size, float *rand, float prob, float scale
		         */ 
				kernelBackParameters = Pointer.to(
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(new int[]{delta.getDataLength()}),
		                Pointer.to(rand.getGpuData()),
		                Pointer.to(new float[]{prob}),
		                Pointer.to(new float[]{scale})
		            );
		        
		        this.N = delta.number;
		        
			}

			cuLaunchKernel(back_function,
		            this.CAFFE_GET_BLOCKS(delta.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelBackParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	 public static void main(String args[]){	
	    	int N = 2;
	    	int C = 1;
	    	int H = 1;
	    	int W = 8;
	    	
	    	float[] x1 = RandomUtils.order(N * C * H * W, 0.0000001f, 0.0000001f);
	    	
	    	float[] x2 = RandomUtils.order(N * C * H * W, 0.0000001f, 0.0000001f);
	    	
	    	float[] d = RandomUtils.order(N * C * H * W, 0.0001f, 0.0001f);
	    	
	    	float[] bias1 = RandomUtils.order(H * W, 0.000001f, 0.00001f);
	    	
	    	float[] bias2 = RandomUtils.order(H * W, 0.000001f, 0.00001f);
	    	
	    	
	    }
	
}
