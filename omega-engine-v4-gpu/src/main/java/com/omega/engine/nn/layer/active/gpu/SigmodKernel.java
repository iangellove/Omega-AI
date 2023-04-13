package com.omega.engine.nn.layer.active.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.driver.CUfunction;

public class SigmodKernel extends BaseKernel{
	
	private CUfunction function;
	
	private CUfunction function_back;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer forwardKernelParameters;
	
	private Pointer backwardKernelParameters;
	
	public SigmodKernel() {
		
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

				function = CUDAModules.getFunctionByModule("H://activeFunction.cu", "sigmod_forward");
				
			}
			
			if(function_back == null) {

				function_back = CUDAModules.getFunctionByModule("H://activeFunction.cu", "sigmod_backward");
				
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
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
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
	
	public void backward(Tensor output,Tensor delta,Tensor diff) {
		
		try {
			
			if(backwardKernelParameters == null) {

		        /**
		         * 设置入参
		         * float* data_im,float* data_col,int n,int height,int width,int kh,int kw,int s,int p,int oh,int ow
		         */ 
				backwardKernelParameters = Pointer.to(
						Pointer.to(output.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(diff.getGpuData()),
		                Pointer.to(new int[]{output.dataLength})
		            );
		        
			}
			
			cuLaunchKernel(function_back,
		            this.CAFFE_GET_BLOCKS(output.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            backwardKernelParameters, null // Kernel- and extra parameters
		        );

//			delta.showDM();
			
//			JCudaDriver.cuCtxSynchronize();
			
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
	    	
	    	SigmodKernel k = new SigmodKernel();

	    	k.forward(input, output);
	    	
	    	k.backward(output, delta, diff);

	    	output.showDM();
	    	
	    	diff.showDM();
	    	
	    	k.backward(output, delta, diff);
	    	
	    	diff.showDM();
	    	
			CUDAMemoryManager.free();
			
	    }
	
	
}
