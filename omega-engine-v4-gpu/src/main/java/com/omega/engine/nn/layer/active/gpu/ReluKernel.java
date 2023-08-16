package com.omega.engine.nn.layer.active.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.active.jobs.relu.ReluBackwardJob;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class ReluKernel extends BaseKernel{
	
	private CUfunction function;
	
	private CUfunction function_back;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	
	private Pointer backwardKernelParameters;
	
	public ReluKernel() {
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

				function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"activeFunction.cu", "relu_forward");
				
			}
			
			if(function_back == null) {

				function_back = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"activeFunction.cu", "relu_backward");
				
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
			
			if(forwardKernelParameters == null || this.N != output.number) {
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
					
			}

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
	
	public void backward(Tensor input,Tensor delta,Tensor diff) {
		
		try {
			
			if(backwardKernelParameters == null) {
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
			}
			
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
	
	public void cpu_backward(Tensor input,Tensor delta,Tensor diff) {
		ReluBackwardJob backward = new ReluBackwardJob(input.data, delta.data, diff.data, 0, diff.dataLength - 1);
		ForkJobEngine.run(backward);
	}
	
	 public static void main(String args[]){	
	    	int N = 128;
	    	int C = 64;
	    	int H = 28;
	    	int W = 28;
	    	
	    	float[] x1 = RandomUtils.order(N * C * H * W, 0.000000000001f, 0.000000000001f);
	    	
	    	float[] bias1 = RandomUtils.order(N * C * H * W, 0.000000000001f, 0.00000000001f);
	    	
	    	Tensor input = new Tensor(N, C, H, W, x1, true);
	    	
	    	Tensor output = new Tensor(N, C, H, W, true);
	    	
	    	Tensor delta = new Tensor(N, C, H, W, bias1, true);
	    	
	    	Tensor diff = new Tensor(N, C, H, W, true);
	    	
	    	Tensor diff_cpu = new Tensor(N, C, H, W);
	    	
	    	ReluKernel k = new ReluKernel();

	    	k.forward(input, output);
	    	
	    	k.backward(input, delta, diff);
	    	
	    	k.cpu_backward(input, delta, diff_cpu);

	    	output.syncHost();
	    	
	    	diff.syncHost();
//
//	    	System.out.println(CheckArrayUtils.check(ouput.data, diff_cpu.data));
//	    	
	    	System.out.println(CheckArrayUtils.check(diff.data, diff_cpu.data));
	    	
			CUDAMemoryManager.free();
			
	    }
	
	
}
