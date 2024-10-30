package com.omega.engine.nn.network.vae;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class VAEKernel extends BaseKernel{
	
	private CUfunction function;
	
	private CUfunction function_back;
	
	private CUfunction kl_loss_function;
	
	private CUfunction kl_loss_function_back;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	
	private Pointer backwardKernelParameters;
	
	public VAEKernel() {
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

				function = CUDAModules.getLocalFunctionByModule("VAE.cu", "reparameterize_forward");
				
			}
			
			if(function_back == null) {

				function_back = CUDAModules.getLocalFunctionByModule("VAE.cu", "reparameterize_backward");
				
			}
			
			if(kl_loss_function == null) {

				kl_loss_function = CUDAModules.getLocalFunctionByModule("VAE.cu", "kl_loss");
				
			}
			
			if(kl_loss_function_back == null) {

				kl_loss_function_back = CUDAModules.getLocalFunctionByModule("VAE.cu", "kl_loss_back");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor mu,Tensor logvar,Tensor eps,Tensor output) {
		
		try {

			/**
	         * 设置入参
	         * float *mu,float *logvar,float *eps, float *output, int n
	         */ 
			forwardKernelParameters = Pointer.to(
	        		Pointer.to(mu.getGpuData()),
	        		Pointer.to(logvar.getGpuData()),
	        		Pointer.to(eps.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[]{output.dataLength})
	            );
			
			this.N = output.number;

			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor delta,Tensor eps,Tensor logvar,Tensor dmu,Tensor dlogvar) {
		
		try {

			/**
	         * 设置入参
	         * float *dmu,float *dlogvar,float *eps,float *logvar, float *delta, int n
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(dmu.getGpuData()),
					Pointer.to(dlogvar.getGpuData()),
					Pointer.to(eps.getGpuData()),
					Pointer.to(logvar.getGpuData()),
	        		Pointer.to(delta.getGpuData()),
	                Pointer.to(new int[]{delta.dataLength})
	            );
			
			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(delta.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void kl(Tensor mu,Tensor logvar,float kl_weight,Tensor output) {
		
		try {

			/**
	         * 设置入参
	         * float *mu,float *logvar,float kl_weight, float *klLoss, int n
	         */ 
			forwardKernelParameters = Pointer.to(
	        		Pointer.to(mu.getGpuData()),
	        		Pointer.to(logvar.getGpuData()),
	        		Pointer.to(new float[] {kl_weight}),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[]{output.dataLength})
	            );
			
			this.N = output.number;

			cuLaunchKernel(kl_loss_function,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            forwardKernelParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void kl_back(Tensor mu,Tensor logvar,float kl_weight,Tensor dmu,Tensor dlogvar) {
		
		try {

			/**
	         * 设置入参
	         * float *mu,float *logvar,float kl_weight, float *dmu, float * dlogvar,int batch, int n
	         */ 
			backwardKernelParameters = Pointer.to(
					Pointer.to(mu.getGpuData()),
					Pointer.to(logvar.getGpuData()),
					Pointer.to(new float[] {kl_weight}),
					Pointer.to(dmu.getGpuData()),
					Pointer.to(dlogvar.getGpuData()),
					Pointer.to(new int[] {mu.number}),
	                Pointer.to(new int[]{mu.dataLength})
	            );
			
			cuLaunchKernel(kl_loss_function_back,
		            this.CAFFE_GET_BLOCKS(mu.dataLength),  1, 1,      // Grid dimension
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
//	    	int N = 5;
//	    	int C = 1;
//	    	int H = 1;
//	    	int W = 8;
//	    	
//	    	float[] x1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
//	    	
//	    	float[] d1 = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
//	    	
//	    	Tensor input = new Tensor(N, C, H, W, x1, true);
//	    	
//	    	Tensor output = new Tensor(N, C, H, W, true);
//	    	
//	    	Tensor delta = new Tensor(N, C, H, W, d1, true);
//	    	
//	    	Tensor diff = new Tensor(N, C, H, W, true);
//	    
//	    	VAEKernel k = new VAEKernel();
//
////	    	k.forward(input, output);
////	    	
////	    	k.backward(input, delta, diff);
//	    	
//	    	output.showDM();
//	    	
//	    	diff.showDM();
//
//			CUDAMemoryManager.free();
			
	    }
	
	
}
