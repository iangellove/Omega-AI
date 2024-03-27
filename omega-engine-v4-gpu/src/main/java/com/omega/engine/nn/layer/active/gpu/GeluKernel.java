package com.omega.engine.nn.layer.active.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

public class GeluKernel extends BaseKernel{
	
	private CUfunction function;
	
	private CUfunction function_back;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	
	private Pointer backwardKernelParameters;
	
	public GeluKernel() {
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

				function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"activeFunction.cu", "gelu_fwd_cuda");
				
			}
			
			if(function_back == null) {

				function_back = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"activeFunction.cu", "gelu_bwd_cuda");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor input,Tensor output) {
		
		try {
			
//			if(forwardKernelParameters == null || this.N != output.number) {
					/**
			         * 设置入参
			         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
			         */ 
					forwardKernelParameters = Pointer.to(
			        		Pointer.to(input.getGpuData()),
			                Pointer.to(output.getGpuData()),
			                Pointer.to(new int[]{output.dataLength})
			            );
					
					this.N = output.number;
					
//			}

			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(input.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

//			JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward(Tensor input,Tensor output,int index,int length) {
		
		try {

			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			forwardKernelParameters = Pointer.to(
	        		Pointer.to(input.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
	                Pointer.to(output.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
	                Pointer.to(new int[]{length})
	            );
			
			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(length),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward(Pointer input,Pointer output,int length) {
		
		try {

			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			forwardKernelParameters = Pointer.to(
	        		Pointer.to(input),
	                Pointer.to(output),
	                Pointer.to(new int[]{length})
	            );

			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(length),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff) {
		
		try {
			
//			if(backwardKernelParameters == null) {
				/**
		         * 设置入参
		         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
		         */ 
				backwardKernelParameters = Pointer.to(
						Pointer.to(input.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(diff.getGpuData()),
		                Pointer.to(new int[]{input.dataLength})
		            );
//			}
			
			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(input.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff,int index,int length) {
		
		try {

			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(input.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
	        		Pointer.to(delta.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
	                Pointer.to(diff.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
	                Pointer.to(new int[]{length})
	            );

			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(length),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Pointer input,Pointer delta,Pointer diff,int length) {
		
		try {

			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(input),
	        		Pointer.to(delta),
	                Pointer.to(diff),
	                Pointer.to(new int[]{length})
	            );

			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(length),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String args[]){	
	    	int N = 5;
	    	int C = 1;
	    	int H = 1;
	    	int W = 8;
	    	
	    	float[] x1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
	    	
	    	float[] d1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
	    	
	    	Tensor input = new Tensor(N, C, H, W, x1, true);
	    	
	    	Tensor output = new Tensor(N, C, H, W, true);
	    	
	    	Tensor delta = new Tensor(N, C, H, W, d1, true);
	    	
	    	Tensor diff = new Tensor(N, C, H, W, true);
	    
	    	GeluKernel k = new GeluKernel();

	    	k.forward(input, output);
	    	
	    	k.backward(input, delta, diff);
	    	
	    	output.showDM();
	    	
	    	diff.showDM();

			CUDAMemoryManager.free();
			
	    }
	
	
}
