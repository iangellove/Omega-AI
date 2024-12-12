package com.omega.engine.nn.layer.active.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;

public class SiLUKernel extends BaseKernel{
	
	private CUfunction function;
	
	private CUfunction function_back;
	
	private CUfunction function_back_temp;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	
	private Pointer backwardKernelParameters;
	
	public SiLUKernel() {
		
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

				function = CUDAModules.getLocalFunctionByModule("activeFunction.cu", "silu_forward");
				
			}
			
			if(function_back == null) {

				function_back = CUDAModules.getLocalFunctionByModule("activeFunction.cu", "silu_backward");
				
			}
			
			if(function_back_temp == null) {

				function_back_temp = CUDAModules.getLocalFunctionByModule("activeFunction.cu", "silu_backward_temp");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
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
	
//	public void forward(Tensor input,Tensor output,int batch,int step) {
//		
//		try {
//
//			/**
//	         * 设置入参
//	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
//	         */ 
//			forwardKernelParameters = Pointer.to(
//	        		Pointer.to(input.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT)),
//	                Pointer.to(output.getGpuData().withByteOffset(step * batch * input.getOnceSize() * Sizeof.FLOAT)),
//	                Pointer.to(new int[]{batch * output.getOnceSize()})
//	            );
//			
//			cuLaunchKernel(function,
//		            this.CAFFE_GET_BLOCKS(batch * input.getOnceSize()),  1, 1,      // Grid dimension
//		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
//		            0, null,               // Shared memory size and stream
//		            forwardKernelParameters, null // Kernel- and extra parameters
//		        );
//
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//		}
//		
//	}
	
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
		                Pointer.to(new int[]{input.dataLength})
		            );
				
				this.N = output.number;
				
//			}
			
			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(input.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor output,Tensor delta,Tensor diff) {
		
		try {
//			System.out.println(delta);
//			if(backwardKernelParameters == null) {

//			delta.showDM();
//			output.showDM();
//			System.out.println(JsonUtils.toJson(delta.getByNumber(0)));
		        /**
		         * 设置入参
		         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
		         */ 
				backwardKernelParameters = Pointer.to(
						Pointer.to(input.getGpuData()),
						Pointer.to(output.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(diff.getGpuData()),
		                Pointer.to(new int[]{output.dataLength})
		            );
		        
//			}
			
			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

//			diff.showDM();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backwardTemp(Tensor input,Tensor output,Tensor delta,Tensor diff) {
		
		try {
//			System.out.println(delta);
//			if(backwardKernelParameters == null) {

//			delta.showDM();
//			output.showDM();
//			System.out.println(JsonUtils.toJson(delta.getByNumber(0)));
		        /**
		         * 设置入参
		         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
		         */ 
				backwardKernelParameters = Pointer.to(
						Pointer.to(input.getGpuData()),
						Pointer.to(output.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(diff.getGpuData()),
		                Pointer.to(new int[]{output.dataLength})
		            );
		        
//			}
			
			cuLaunchKernel(function_back_temp,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

//			diff.showDM();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor output,Tensor delta,Tensor diff,int index,int length) {
		
		try {
			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(output.getGpuData().withByteOffset(index * Sizeof.FLOAT)),
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
	
	public void backward(Tensor input,Tensor delta,Tensor diff,int step) {
		
		try {

			/**
	         * 设置入参
	         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(input.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)),
	        		Pointer.to(delta.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)),
	                Pointer.to(diff.getGpuData().withByteOffset(step * input.getOnceSize() * Sizeof.FLOAT)),
	                Pointer.to(new int[]{input.getOnceSize()})
	            );

			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(input.getOnceSize()),  1, 1,      // Grid dimension
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
	    	int N = 2;
	    	int C = 1;
	    	int H = 1;
	    	int W = 8;
	    	
	    	float[] x1 = new float[] {1,2,3,4,-5,6,-7,-8,9,10,11,-12,13,14,15,-16};
	    	
	    	float[] bias1 = MatrixUtils.one(x1.length);
	    	
	    	Tensor input = new Tensor(N, C, H, W, x1, true);
	    	
	    	Tensor output = new Tensor(N, C, H, W, true);
	    	
	    	Tensor delta = new Tensor(N, C, H, W, bias1, true);
	    	
	    	Tensor diff = new Tensor(N, C, H, W, true);
	    	
	    	SiLUKernel k = new SiLUKernel();

	    	k.forward(input, output);
	    	
	    	k.backward(input ,output, delta, diff);

	    	output.showDM();
	    	
	    	diff.showDM();
	    	
	    	k.backward(input, output, delta, diff);
	    	
	    	diff.showDM();
	    	
			CUDAMemoryManager.free();
			
	 }
	
	
}
