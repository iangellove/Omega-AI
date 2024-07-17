package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcudnn.JCudnn;
import jcuda.runtime.cudaError;

public class AttentionKernel extends BaseKernel{

	private CUfunction permute_function;
	
	private CUfunction unpermute_function;
	
	private CUfunction permute_backward_function;
	
	private CUfunction unpermute_backward_function;
	
	private CUfunction softmax_forward_function;
	
	private CUfunction softmax_test_forward_function;
	
	private CUfunction softmax_backward_function;

	private CUfunction scale_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private int BLOCK = 512;
	
	private Pointer permuteKernelParameters;
	
	private Pointer unpermuteKernelBackParameters;
	
	private Pointer permute_backwardKernelParameters;
	
	private Pointer unpermute_backwardKernelBackParameters;
	
	private Pointer softmaxForwardParameters;
	
	private Pointer softmaxBackwardParameters;
	
	private Pointer scaleParameters;
	
	public AttentionKernel() {
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

			if(permute_function == null) {

				permute_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "permute_kernel");
				
			}
			
			if(unpermute_function == null) {

				unpermute_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "unpermute_kernel");
				
			}
			
			if(permute_backward_function == null) {

				permute_backward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "permute_kernel_backward");
				
			}
			
			if(unpermute_backward_function == null) {

				unpermute_backward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "unpermute_kernel_backward");
				
			}

			if(softmax_forward_function == null) {
//				softmax_forward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel"); 
				softmax_forward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel5");
			}
			
			if(softmax_test_forward_function == null) {
//				softmax_forward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel"); 
				softmax_test_forward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_forward_kernel4");
			}
			
			if(softmax_backward_function == null) {
//				softmax_backward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_kernel");
				softmax_backward_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "softmax_autoregressive_backward_kernel7");
			}
			
			if(scale_function == null) {
				scale_function = CUDAModules.getLocalFunctionByModule("AttentionKernel.cu", "scale_kernel");
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void scale(Tensor input,float scale,int B,int NH,int T) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* inp, float scale, int B, int NH, int T
	         */ 
			scaleParameters = Pointer.to(
	        		Pointer.to(input.getGpuData()),
	        		Pointer.to(new float[]{scale}),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{NH}),
	                Pointer.to(new int[]{T})
	            );
	        
			int total_threads = B * NH * T * T;
		    int num_blocks = get_number_of_blocks(total_threads, BLOCK);

		    checkCUDA(cuLaunchKernel(scale_function,
					num_blocks,  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            scaleParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void permute(Tensor input,Tensor query,Tensor key,Tensor value,int B,int N,int NH,int d) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* q, float* k, float* v, const float* inp,int B, int N, int NH, int d
	         */ 
			permuteKernelParameters = Pointer.to(
	        		Pointer.to(query.getGpuData()),
	        		Pointer.to(key.getGpuData()),
	        		Pointer.to(value.getGpuData()),
	        		Pointer.to(input.getGpuData()),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{NH}),
	                Pointer.to(new int[]{d})
	            );
	        
			int total_threads = B * NH * N * d;
		    int num_blocks = get_number_of_blocks(total_threads, BLOCK);
			
		    checkCUDA(cuLaunchKernel(permute_function,
					num_blocks,  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            permuteKernelParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void permute_backward(Tensor diff,Tensor dQuery,Tensor dKey,Tensor dValue,int B,int N,int NH,int d) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* dinp, const float* dq, const float* dk, const float* dv,int B, int N, int NH, int d
	         */ 
			permute_backwardKernelParameters = Pointer.to(
	        		Pointer.to(diff.getGpuData()),
	        		Pointer.to(dQuery.getGpuData()),
	        		Pointer.to(dKey.getGpuData()),
	        		Pointer.to(dValue.getGpuData()),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{NH}),
	                Pointer.to(new int[]{d})
	            );
	        
			int total_threads = B * NH * N * d;
		    int num_blocks = get_number_of_blocks(total_threads, BLOCK);
			
		    checkCUDA(cuLaunchKernel(permute_backward_function,
					num_blocks,  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            permute_backwardKernelParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void unpermute(Tensor input,Tensor output,int B,int N,int NH,int d) {
		
		try {
			
	        /**
	         * 设置入参
	         * const float* inp, float *out, int B, int N, int NH, int d
	         */ 
			unpermuteKernelBackParameters = Pointer.to(
	        		Pointer.to(input.getGpuData()),
	        		Pointer.to(output.getGpuData()),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{NH}),
	                Pointer.to(new int[]{d})
	            );
	        
			int total_threads = output.getDataLength();
		    int num_blocks = get_number_of_blocks(total_threads, BLOCK);
			
		    checkCUDA(cuLaunchKernel(unpermute_function,
					num_blocks,  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            unpermuteKernelBackParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void unpermute_backward(Tensor din,Tensor dout,int B,int N,int NH,int d) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* dinp, const float *dout, int B, int N, int NH, int d
	         */ 
			unpermute_backwardKernelBackParameters = Pointer.to(
	        		Pointer.to(din.getGpuData()),
	        		Pointer.to(dout.getGpuData()),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{N}),
	                Pointer.to(new int[]{NH}),
	                Pointer.to(new int[]{d})
	            );
	        
			int total_threads = B * N * NH * d;
		    int num_blocks = get_number_of_blocks(total_threads, BLOCK);
			
		    checkCUDA(cuLaunchKernel(unpermute_backward_function,
					num_blocks,  1, 1,      // Grid dimension
		            BLOCK, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            unpermute_backwardKernelBackParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	/**
	 * N = B * NH
	 * @param input
	 * @param output
	 * @param scale
	 * @param B
	 * @param T
	 */
	public void softmax_forward(Tensor input,Tensor output,int B,int NH,int T,float scale) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* out, float inv_temperature, const float* inp, int N, int T
	         * float* out, const float* inp, int N, int C
	         */ 
			softmaxForwardParameters = Pointer.to(
	        		Pointer.to(output.getGpuData()),
	        	    Pointer.to(new float[]{scale}),
	        	    Pointer.to(input.getGpuData()),
	                Pointer.to(new int[]{B * NH}),
	                Pointer.to(new int[]{T})
	            );
	        
//			int softmax_block_size = 256;
//		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
			int softmax_block_size = 256;
//		    int grid_size = B * NH * T;
		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
//		    int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
			
		    checkCUDA(cuLaunchKernel(softmax_forward_function,
		    		grid_size,  1, 1,      // Grid dimension
		    		softmax_block_size, 1, 1,      // Block dimension
		    		0, null,               // Shared memory size and stream
		            softmaxForwardParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	/**
	 * N = B * NH
	 * @param input
	 * @param output
	 * @param scale
	 * @param B
	 * @param T
	 */
	public void softmax_test_forward(Tensor input,Tensor output,int B,int NH,int T) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* out,float scale, const float* inp, int N, int C
	         */ 
			softmaxForwardParameters = Pointer.to(
	        		Pointer.to(output.getGpuData()),
//	        	    Pointer.to(new float[]{scale}),
	        	    Pointer.to(input.getGpuData()),
	                Pointer.to(new int[]{B * NH * T}),
	                Pointer.to(new int[]{T})
	            );
	        
//			int softmax_block_size = 256;
//		    int grid_size = get_number_of_blocks(B * NH * T * 32, softmax_block_size);
			int softmax_block_size = 256;
		    int grid_size = B * NH * T;
//		    int grid_size = (int) Math.ceil(B * NH * T * 32 / softmax_block_size);
		    int shared_mem_size = 2 * softmax_block_size / 32 * Sizeof.FLOAT;
			
		    checkCUDA(cuLaunchKernel(softmax_test_forward_function,
		    		grid_size,  1, 1,      // Grid dimension
		    		softmax_block_size, 1, 1,      // Block dimension
		    		shared_mem_size, null,               // Shared memory size and stream
		            softmaxForwardParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	/**
	 * N = B * NH
	 * @param input
	 * @param output
	 * @param scale
	 * @param B
	 * @param T
	 */
	public void softmax_backward(Tensor dpreatt,Tensor datt,Tensor att,int B,int T,int C,int NH,float scale) {
		
		try {
			
	        /**
	         * 设置入参
	         * float* dpreatt, const float* datt, const float* att,
               int B, int T, int C, float scale, int BlockSize
	         */ 
			int block_size = 256;
			softmaxBackwardParameters = Pointer.to(
	        		Pointer.to(dpreatt.getGpuData()),
	        		Pointer.to(datt.getGpuData()),
	        		Pointer.to(att.getGpuData()),
	                Pointer.to(new int[]{B}),
	                Pointer.to(new int[]{T}),
	                Pointer.to(new int[]{C}),
	                Pointer.to(new float[]{scale})
	            );
//			int num_blocks = get_number_of_blocks(32/8*T, block_size);
//	        int num_blocks = get_number_of_blocks(T, block_size);
		    checkCUDA(cuLaunchKernel(softmax_backward_function,
		    		T,  B * NH, 1,      // Grid dimension
		    		block_size, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            softmaxBackwardParameters, null // Kernel- and extra parameters
		        ));

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public int get_number_of_blocks(int array_size, int block_size)
	{
		return array_size / block_size + ((array_size % block_size > 0) ? 1 : 0);
	}
	

	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	
}
