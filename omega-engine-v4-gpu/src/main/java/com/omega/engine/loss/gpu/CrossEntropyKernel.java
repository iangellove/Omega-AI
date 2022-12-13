package com.omega.engine.loss.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class CrossEntropyKernel extends BaseKernel {
	
	private CUfunction loss_function;
	
	private CUfunction nl_loss_function;
	
	private CUfunction log_softmax_nl_loss_function;
	
	private CUfunction check_function;
	
	private CUfunction loss_backward_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer kernelParameters;
	
	private Pointer log_softmax_nl_loss_kernelParameters;
	
	private Pointer checkParameters;
	
	private Pointer backKernelParameters;
	
	public CrossEntropyKernel() {
		init();
	}
	
	public void initFunction() {
		
		try {

			if(loss_function == null) {
				
				loss_function = CUDAModules.getFunctionByModule("H://CrossEntropyKernel.cu", "loss");
        
			}
			
			if(nl_loss_function == null) {
				
				nl_loss_function = CUDAModules.getFunctionByModule("H://CrossEntropyKernel.cu", "nl_loss");
        
			}
			
			if(log_softmax_nl_loss_function == null) {
				
				log_softmax_nl_loss_function = CUDAModules.getFunctionByModule("H://CrossEntropyKernel.cu", "log_softmax_nl_loss");
        
			}
			
			if(check_function == null) {
				
				check_function = CUDAModules.getFunctionByModule("H://CrossEntropyKernel.cu", "check");
        
			}
			
			if(loss_backward_function == null) {
				
				loss_backward_function = CUDAModules.getFunctionByModule("H://CrossEntropyKernel.cu", "loss_back2");
        
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

	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor input,Tensor currentLabel,Tensor output) {
		
		if(log_softmax_nl_loss_kernelParameters == null || this.N != output.number) {
			/**
			 * float *input, float *label, float *output, int batch, int n
			 */
			log_softmax_nl_loss_kernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
			this.N = output.number;
			
		}
		
		cuLaunchKernel(log_softmax_nl_loss_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            log_softmax_nl_loss_kernelParameters, null // Kernel- and extra parameters
	        );
		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void forwardCheck(Tensor input,Tensor currentLabel,Tensor output) {
		
//		if(checkParameters == null || this.N != output.number) {
			/**
			 * float *input, float *output, int batch, int n, float temp
			 */
			checkParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
			this.N = output.number;
			
//		}
		
		cuLaunchKernel(check_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            checkParameters, null // Kernel- and extra parameters
	        );
		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void backward(Tensor input,Tensor currentLabel,Tensor diff) {

		if(backKernelParameters == null) {

			/**
			 * float *input, float *currentLabel, float *diff, int n, int batch
			 */
			backKernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );

		}
		
		cuLaunchKernel(loss_backward_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            backKernelParameters, null // Kernel- and extra parameters
	        );
		
//		JCudaDriver.cuCtxSynchronize();
		
	}

	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
}
