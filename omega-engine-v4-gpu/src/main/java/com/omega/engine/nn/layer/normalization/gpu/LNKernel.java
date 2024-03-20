package com.omega.engine.nn.layer.normalization.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.normalization.BNType;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

/**
 * mean = batch均值    : 1/n∑xi
 * var = batch方差    : 1/n∑(xi - mean)^2
 * std = sqrt(var + eta)
 * xhati = (xi - mean) / std
 * yi = gama * xhati + beta
 * dgama = ∑delta * xhat
 * dbeta = ∑delta
 * dxhati = gama * deltai
 * dxi = 1 / std * (dxhati - mean(dxhat) - xhati * mean(dxhat * xhat))
 */
public class LNKernel extends BaseKernel{
	
	public BNType bnType = null;
	
	private int B;
	private int W;
	
	private int slice_size;
	
	/**
	 * 向前方法
	 */
	private CUfunction forward_small_function;
	private CUfunction forward_1_function;
	private CUfunction forward_2_function;
	private CUfunction forward_3_function;
	private CUfunction forward_4_function;
	private CUfunction forward_5_function;
	
	private CUfunction forward_test_function;
	
	/**
	 * 反向传播方法
	 */
	private CUfunction backward_small_function;
	private CUfunction backward_1_function;
	private CUfunction backward_2_function;
	private CUfunction backward_3_function;
	private CUfunction backward_4_function;
	private CUfunction backward_5_function;
	
	private CUfunction backward_ig_function;
	private CUfunction backward_fp_function;
	private CUfunction backward_input_function;
	private CUfunction backward_gamma_function;
	private CUfunction backward_gamma_simple_function;
	
	private CUfunction backward_input_function2;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private int MAX_GRID_SIZE = 480;
	
	private float eta = 1e-6f;
	
	private int warp_size = 32;
	
	private int kCUDABlockReduceNumThreads = 512;
	private int kCUDANumThreads = 256;
	private int kColwiseReduceTileSize = 32;
	
	/**
	 * 前向方法参数
	 */
	private Pointer forwardSmallParameters;
	private Pointer forwardParameters;
	private Pointer forwardTestParameters;
	
	/**
	 * 反向方法参数
	 */
	private Pointer backwardSmallParameters;
	private Pointer backwardParameters;
	
	private Pointer backwardIGParameters;
	private Pointer backwardFGParameters;
	private Pointer backwardInputParameters;
	private Pointer backwardGammaParameters;
	private Pointer backwardGammaSampleParameters;
	
	private Pointer backwardInputParameters2;
	
	private CUdeviceptr d_mean;
	private CUdeviceptr d_var;
	
	private CUdeviceptr d_s;
	private CUdeviceptr d_b;
	private CUdeviceptr d_scale;
	private CUdeviceptr d_bias;
	
	
	public LNKernel(int W,BNType bnType) {
		this.W = W;
		this.bnType = bnType;
		if (this.W<=warp_size){
	        int tmp_depth = this.W;
	        int slice_size = 1;
	        while((tmp_depth >>= 1) != 0)slice_size*=2;
	        this.slice_size = slice_size>=this.W?slice_size:slice_size*2;
	    }else{
	        int slice_size = (this.W/warp_size)*warp_size;
	        this.slice_size = slice_size>=this.W?slice_size:slice_size+warp_size;
	    }
		init();
	}
	
	private void initKernel() {
		/**
		 * 申请向前传播参数显存
		 */
		this.d_mean = CUDAMemoryManager.getDevice(B);
		this.d_var = CUDAMemoryManager.getDevice(B);
		this.d_s = CUDAMemoryManager.getDevice(B);
		this.d_b = CUDAMemoryManager.getDevice(B);
		this.d_scale = CUDAMemoryManager.getDevice(B);
		this.d_bias = CUDAMemoryManager.getDevice(B);
	}
	
	public void initFunction() {
		
		try {
			
			if(forward_small_function == null) {
				forward_small_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNormFusedSmallGPUKernel");
			}
			
			if(forward_1_function == null) {
				forward_1_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNorm1GPUKernel");
			}
			
			if(forward_2_function == null) {
				forward_2_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNorm2GPUKernel");
			}
			
			if(forward_3_function == null) {
				forward_3_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNorm3GPUKernel");
			}
			
			if(forward_4_function == null) {
				forward_4_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNorm4GPUKernel");
			}
			
			if(forward_5_function == null) {
				forward_5_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel.cu", "LayerNorm5GPUKernel");
			}
			
			if(backward_small_function == null) {
				backward_small_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNormFusedSmallBackpropGPUKernel");
			}
			
			if(backward_1_function == null) {
				backward_1_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNorm1FusedBackpropGPUKernel");
			}
			
			if(backward_2_function == null) {
				backward_2_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNorm2FusedBackpropGPUKernel");
			}
			
			if(backward_3_function == null) {
				backward_3_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNorm3FusedBackpropGPUKernel");
			}
			
			if(backward_4_function == null) {
				backward_4_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNorm4FusedBackpropGPUKernel");
			}
			
			if(backward_5_function == null) {
				backward_5_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernelBackward.cu", "LayerNorm5FusedBackpropGPUKernel");
			}
			
			if(forward_test_function == null) {
				forward_test_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "LayerNormFusedForwardKernel");
			}

			if(backward_ig_function == null) {
				backward_ig_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "ComputeInternalGradientsCUDAKernel");
			}
			
			if(backward_fp_function == null) {
				backward_fp_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "ComputeGradientFusedParamsCUDAKernel");
			}
			
			if(backward_input_function == null) {
				backward_input_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "layer_norm_grad_input_kernel");
			}
			
			if(backward_gamma_function == null) {
				backward_gamma_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "GammaBetaBackwardCUDAKernel");
			}
			
			if(backward_gamma_simple_function == null) {
				backward_gamma_simple_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel3.cu", "GammaBetaBackwardSimpleCUDAKernel");
			}
			
			if(backward_input_function2 == null) {
				backward_input_function2 = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"LNKernel4.cu", "aten_layer_norm_grad_input_kernel");
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
	
	public boolean checkBatch(Tensor input) {
		int batchSize = 0;
		switch (bnType) {
		case fully_bn:
			batchSize = input.number * input.channel * input.height;
//			System.out.println("batchSize:"+batchSize);
//			System.out.println("B:"+B);
			break;
		case conv_bn:
			batchSize = input.number * input.channel;
			break;
		}
		if(B != batchSize){
			this.B = batchSize;
			return false;
		}
		return true;
	}
	
	public void initForwardSmall(Tensor input,Tensor gamma,Tensor beta,Tensor output,int num_blocks,int slice_per_block) {
		
		if(checkBatch(input)) {
			
			/**
			 * const int slice_size,const int in_depth,const int n_inputs,const float epsilon, const float* __restrict__ input,
    const float* __restrict__ gamma, const float* __restrict__ beta,
    float* __restrict__ output, const int num_blocks, const int slice_per_block
			 */
			forwardSmallParameters = Pointer.to(
					Pointer.to(new int[] {slice_size}),
					Pointer.to(new int[] {W}),
					Pointer.to(new int[] {input.getDataLength()}),
					Pointer.to(new float[] {eta}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {num_blocks}),
	                Pointer.to(new int[] {slice_per_block})
	            );

		}
		
	}
	
	public void initTestForward(Tensor input,Tensor gamma,Tensor beta,Tensor output) {
	
		if(forwardTestParameters == null) {

			initKernel();

			/**
			 * int N,
	        float eps,
	        float const *X,
	        float *mean,
	        float *rstd,
	        float const *gamma,
	        float const *beta,
	        float *Y
			 */
			forwardTestParameters = Pointer.to(
					Pointer.to(new int[] {W}),
					Pointer.to(new float[] {eta}),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(output.getGpuData())
	            );

		}
		
	}
	
	public void initForward(Tensor input,Tensor gamma,Tensor beta,Tensor output,int num_blocks) {
		
		if(checkBatch(input)) {
			
			/**
			 * const int slice_size,const int in_depth,const int n_inputs,const float epsilon,
                                   const float* __restrict__ input,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ beta,
                                   float* __restrict__ output,
                                   const int num_blocks
			 */
			forwardParameters = Pointer.to(
					Pointer.to(new int[] {slice_size}),
					Pointer.to(new int[] {W}),
					Pointer.to(new int[] {input.getDataLength()}),
					Pointer.to(new float[] {eta}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {num_blocks})
	            );
	
		}
		
	}
	
	public void initBackwardSmall(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta,int num_blocks,int slice_per_block) {

		if(backwardSmallParameters == null) {
			
			/**
			 * const int slice_size,const int in_depth,const int n_inputs,const float epsilon, const float* __restrict__ input,
		    const float* __restrict__ out_back, const float* __restrict__ gamma,
		    float* __restrict__ in_back, float* __restrict__ gamma_back,
		    float* __restrict__ beta_back, const int num_blocks,
		    const int slice_per_block
			 */
			backwardSmallParameters = Pointer.to(
					Pointer.to(new int[] {slice_size}),
					Pointer.to(new int[] {W}),
					Pointer.to(new int[] {input.getDataLength()}),
					Pointer.to(new float[] {eta}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(delta.getGpuData()),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(dgamma.getGpuData()),
	                Pointer.to(dbeta.getGpuData()),
	                Pointer.to(new int[] {num_blocks}),
	                Pointer.to(new int[] {slice_per_block})
	            );
			
		}
		
	}
	
	public void initBackwardTest(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
		if(backwardInputParameters == null) {

			/**
			 * int N, float const *dY, float const *X, float const *gamma, float *ds, float *db
			 */
			backwardIGParameters = Pointer.to(
					Pointer.to(new int[] {W}),
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(gamma.getGpuData()), 
	                Pointer.to(d_s),
	                Pointer.to(d_b)
	            );

			/**
			 * int M,int N,float const *mean,float const *rstd,float const *ds,float const *db,float *c1,float *c2
			 */
			backwardFGParameters = Pointer.to(
					Pointer.to(new int[] {B}),
					Pointer.to(new int[] {W}),
					Pointer.to(d_mean),
					Pointer.to(d_var),
					Pointer.to(d_s),
		            Pointer.to(d_b),
		            Pointer.to(d_scale),
					Pointer.to(d_bias) 
	            );
			
			/**
			 * float const *__restrict__ dY,
               float const *__restrict__ X,
               float const *__restrict__ mean,
               float const *__restrict__ rstd,
               float const *__restrict__ gamma,
               float *dX,
               int const N
			 */
			backwardInputParameters = Pointer.to(
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
		            Pointer.to(gamma.getGpuData()),
		            Pointer.to(diff.getGpuData()),
		            Pointer.to(new int[] {W})
	            );
			
			/**
			 * int M,
               int N,
               float const *dY,
               float const *X,
               float const *mean,
               float const *rstd,
               float *dg,
               float *db
			 */
			backwardGammaSampleParameters = Pointer.to(
					Pointer.to(new int[] {B}),
					Pointer.to(new int[] {W}),
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
		            Pointer.to(dgamma.getGpuData()),
		            Pointer.to(dbeta.getGpuData())
	            );
			
			/**
			 * int M,
               int N,
               float const *dY,
               float const *X,
               float const *mean,
               float const *rstd,
               float *dg,
               float *db
			 */
			backwardGammaParameters = Pointer.to(
					Pointer.to(new int[] {B}),
					Pointer.to(new int[] {W}),
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
		            Pointer.to(dgamma.getGpuData()),
		            Pointer.to(dbeta.getGpuData())
	            );
			
		}
		
	}
	
	public void initBackward3(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {

		if(backwardInputParameters2 == null) {

			/**
			 * aten_layer_norm_grad_input_kernel(
			   const float* __restrict__ dY,
			   const float* __restrict__ X,
			   const float* __restrict__ mean,
			   const float* __restrict__ rstd,
			   const float* __restrict__ gamma,
			   float*  dX,
			   const int N
			 */
			System.out.println(W);
			backwardInputParameters2 = Pointer.to(
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
					Pointer.to(gamma.getGpuData()), 
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {W})
	            );
			
		}
		
	}
	
	public void initBackward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta,int num_blocks) {

		if(backwardParameters == null) {
			
			/**
			 * const int in_depth,
			 * const int n_inputs,
			 * const float epsilon,
			 * const float* __restrict__ input,
			 * const float* __restrict__ out_back,
			 * const float* __restrict__ gamma,
			 * float* __restrict__ in_back,
			 * float* __restrict__ gamma_back,
			 * float* __restrict__ beta_back, 
			 * const int num_blocks
			 */
			backwardParameters = Pointer.to(
					Pointer.to(new int[] {W}),
					Pointer.to(new int[] {input.getDataLength()}),
					Pointer.to(new float[] {eta}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(delta.getGpuData()),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(dgamma.getGpuData()),
	                Pointer.to(dbeta.getGpuData()),
	                Pointer.to(new int[] {num_blocks})
	            );
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor gamma, Tensor beta, Tensor input, Tensor output) {

		try {
			
			boolean check = checkBatch(input);
			
			if (slice_size <= warp_size) {
			      int block_size = 256;
			      int slice_per_block = block_size / slice_size;
			      int num_blocks = get_num_blocks(B, slice_per_block);
			      int grid_size = Math.min(120, num_blocks);
			      
			      if(!check) {
			    	 initForwardSmall(input, gamma, beta, output, num_blocks, slice_per_block);
			      }		 
			      
			      cuLaunchKernel(forward_small_function,
			        		grid_size,  1, 1,      // Grid dimension
			        		block_size, 1, 1,      // Block dimension
				            0, null,               // Shared memory size and stream
				            forwardSmallParameters, null // Kernel- and extra parameters
				        );
			      
			}else {

				int[] bs = get_block_size(slice_size);
				int block_size = bs[0];
			    int mult = bs[1];
			    int num_blocks = B;
			    int sbytes = 2 * Sizeof.FLOAT;
			    int grid_size = Math.min(MAX_GRID_SIZE, num_blocks);
			    
			    if(!check) {
			    	initForward(input, gamma, beta, output, num_blocks);
			    }
			    
			    CUfunction func = null;
			    
			    switch (mult) {
				case 1:
					func = forward_1_function;
					break;
				case 2:
					func = forward_2_function;
					break;
				case 3:
					func = forward_3_function;
					break;
				case 4:
					func = forward_4_function;
					break;
				case 5:
					func = forward_5_function;
					break;
				}
			    
			    cuLaunchKernel(func,
			    		grid_size,  1, 1,      // Grid dimension
			    		block_size, 1, 1,      // Block dimension
			    		sbytes, null,               // Shared memory size and stream
			            forwardParameters, null // Kernel- and extra parameters
			        );
			    
			} 
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward2(Tensor gamma, Tensor beta, Tensor input, Tensor output) {

		try {
			
			boolean check = checkBatch(input);
			
			if(!check) {

				initKernel();

				/**
				 * int N,
		        float eps,
		        float const *X,
		        float *mean,
		        float *rstd,
		        float const *gamma,
		        float const *beta,
		        float *Y
				 */
				forwardTestParameters = Pointer.to(
						Pointer.to(new int[] {W}),
						Pointer.to(new float[] {eta}),
						Pointer.to(input.getGpuData()),
						Pointer.to(d_mean),
						Pointer.to(d_var),
		                Pointer.to(gamma.getGpuData()),
		                Pointer.to(beta.getGpuData()),
		                Pointer.to(output.getGpuData())
		            );

			}
			
			int[] kernel1_parallelism = new int[] {B, 512};
			
			int[] kernel2_parallelism = new int[] {B, 256};
			
		    int num_blocks = Math.max(kernel1_parallelism[0], kernel2_parallelism[0]);
		    
		    int num_threads = Math.max(kernel1_parallelism[1], kernel2_parallelism[1]);
		    
			cuLaunchKernel(forward_test_function,
					num_blocks, 1, 1,      // Grid dimension
					num_threads, 1, 1,      // Block dimension
	        		0, null,               // Shared memory size and stream
		            forwardTestParameters, null // Kernel- and extra parameters
				);
		            
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
		try {
			
			if (slice_size <= warp_size) {
				int block_size = 128;
			    int slice_per_block = block_size / slice_size;
			    int num_blocks = get_num_blocks(B, slice_per_block);
			    int grid_size = Math.min(120, num_blocks);
			    int sbytes = (2 * W) * Sizeof.FLOAT;
				
			    initBackwardSmall(input, delta, diff, gamma, dgamma, dbeta, num_blocks, slice_per_block);
			    
			    cuLaunchKernel(backward_small_function,
			    		grid_size,  1, 1,      // Grid dimension
			    		block_size, 1, 1,      // Block dimension
			    		sbytes, null,               // Shared memory size and stream
			            backwardSmallParameters, null // Kernel- and extra parameters
			        );
			    
			}else {
				
			    int[] bs = get_block_size(slice_size);
			    int block_size = bs[0];
			    int mult = bs[1];
			    int num_blocks = B;
			    int sbytes = 4 * Sizeof.FLOAT;
			    int max_grid = B < 2 * MAX_GRID_SIZE ? 60 : MAX_GRID_SIZE;
			    int grid_size = Math.min(max_grid, num_blocks);
				
			    initBackward(input, delta, diff, gamma, dgamma, dbeta, num_blocks);
			    
			    CUfunction func = null;
//			    System.out.println(grid_size);
//			    System.out.println(mult);
			    switch (mult) {
				case 1:
					func = backward_1_function;
					break;
				case 2:
					func = backward_2_function;
					break;
				case 3:
					func = backward_3_function;
					break;
				case 4:
					func = backward_4_function;
					break;
				case 5:
					func = backward_5_function;
					break;
				}

			    cuLaunchKernel(func,
			    		grid_size,  1, 1,      // Grid dimension
			    		block_size, 1, 1,      // Block dimension
			    		sbytes, null,               // Shared memory size and stream
			            backwardParameters, null // Kernel- and extra parameters
			        );
			    
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward2(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
		try {
			
			initBackwardTest(input, delta, diff, gamma, dgamma, dbeta);
			
			int M = B;
			int N = W;
			
			cuLaunchKernel(backward_ig_function,
		    		M,  1, 1,      // Grid dimension
		    		kCUDABlockReduceNumThreads, 1, 1,      // Block dimension
		    		0, null,               // Shared memory size and stream
		            backwardIGParameters, null // Kernel- and extra parameters
		        );
			
			int B2 = (M + kCUDANumThreads - 1) / kCUDANumThreads;
			cuLaunchKernel(backward_fp_function,
		    		B2,  1, 1,      // Grid dimension
		    		kCUDANumThreads, 1, 1,      // Block dimension
		    		0, null,               // Shared memory size and stream
		            backwardFGParameters, null // Kernel- and extra parameters
		        );
			
			int warp_size = 32;
			int num_threads = 128;
			int nshared = (num_threads / warp_size) * Sizeof.FLOAT;
			cuLaunchKernel(backward_input_function,
		    		M,  1, 1,      // Grid dimension
		    		num_threads, 1, 1,      // Block dimension
		    		nshared, null,               // Shared memory size and stream
		            backwardInputParameters, null // Kernel- and extra parameters
		        );
			
			if (M < 512) {
				int B3 = (N + kCUDANumThreads - 1) / kCUDANumThreads;
				cuLaunchKernel(backward_gamma_simple_function,
			    		B3,  1, 1,      // Grid dimension
			    		kCUDANumThreads, 1, 1,      // Block dimension
			    		0, null,               // Shared memory size and stream
			            backwardGammaSampleParameters, null // Kernel- and extra parameters
			        );
			}else {
				int B4 = (N + kColwiseReduceTileSize - 1) / kColwiseReduceTileSize;
				int kThreadX = kColwiseReduceTileSize;
				int kThreadY = kColwiseReduceTileSize / 2;
				cuLaunchKernel(backward_gamma_function,
						B4,  1, 1,      // Grid dimension
						kThreadX, kThreadY, 1,      // Block dimension
			    		0, null,               // Shared memory size and stream
			            backwardGammaParameters, null // Kernel- and extra parameters
			        );
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward3(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
		try {
			
			initBackward3(input, delta, diff, gamma, dgamma, dbeta);
			
			int M = B;
			int N = W;
			System.out.println("M:"+M);
			int num_threads = 128;
			int nshared = (num_threads / warp_size) * Sizeof.FLOAT;
			
			cuLaunchKernel(backward_input_function2,
					M,  1, 1,      // Grid dimension
					num_threads, 1, 1,      // Block dimension
					nshared, null,               // Shared memory size and stream
		    		backwardInputParameters2, null // Kernel- and extra parameters
		        );
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int get_num_blocks(int n_slices, int slice_per_block) {
		int _num_blocks = n_slices / slice_per_block;
		if (_num_blocks * slice_per_block == n_slices) {
		    return _num_blocks;
		}else {
			return _num_blocks + 1;
		}
	}
	
	public int[] get_block_size(int slice_size) {
		  int block_size = 0;
		  int mult = 0;
		  int _warp_size = 32;
		  int _block_size = _warp_size;
		  int _mult = slice_size / _block_size;
		  mult = _mult * _block_size >= slice_size ? _mult : _mult + 1;
		  while (mult > 5) {
		    _block_size += _warp_size;
		    _mult = slice_size / _block_size;
		    mult = _mult * _block_size >= slice_size ? _mult : _mult + 1;
		  }
		  block_size = _block_size;
		  return new int[] {block_size, mult};
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}

    public void showDM(String id,CUdeviceptr d,float[] data) {
    	JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
    	System.out.println(id + ":"+JsonUtils.toJson(data));
    }
    
    public static void main(String[] args) {
    	
    	int N = 5;
    	int T = 12;
    	int W = 8;
    	
    	float[] data = RandomUtils.order(N * T * W, 0.1f, 0.1f);
    	
//    	float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
//    	float[] betaData = RandomUtils.order(W, 0.1f, 0.0f);
    	
//    	float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
//    	float[] betaData = RandomUtils.order(W, 0.1f, 0.0f);
    	float[] gammaData = MatrixUtils.one(W);
    	float[] betaData = new float[W];
    	
    	Tensor gamma = new Tensor(1, 1, 1, W, gammaData, true);
    	Tensor beta = new Tensor(1, 1, 1, W, betaData, true);
    	
    	Tensor input = new Tensor(N, T, 1, W, data, true);
    	
    	Tensor output = new Tensor(N, T, 1, W, true);
    	
    	Tensor output2 = new Tensor(N, T, 1, W, true);

    	Tensor delta = new Tensor(N, T, 1, W, MatrixUtils.one(N * T * W), true);
    	
    	LNKernel kernel = new LNKernel(W, BNType.fully_bn);

    	Tensor diff2 = new Tensor(N, T, 1, W, true);
    	
    	Tensor dgamma2 = new Tensor(1, 1, 1, W, true);
    	Tensor dbeta2 = new Tensor(1, 1, 1, W, true);
    	
//    	kernel.forward(gamma, beta, input, output);

//    	output.showShape();
//    	output.showDM();
    	for(int i = 0;i<10;i++) {

        	System.out.println("output:");
        	
    		kernel.forward2(gamma, beta, input, output2);
        	output2.showDM();

        	kernel.backward2(input, delta, diff2, gamma, dgamma2, dbeta2);
        	diff2.showDM();
        	dgamma2.showDM();
        	dbeta2.showDM();
        	
        	float[] rmd = RandomUtils.gaussianRandom(N * T * W, 0.1f);
        	Tensor rm = new Tensor(N, T, 1, W, rmd, true);
        	TensorOP.add(input, rm, input);
        	input.showDM();
        	System.out.println("========================");
    	}
    	

//    	Tensor dgamma = new Tensor(1, 1, 1, W, true);
//    	Tensor dbeta = new Tensor(1, 1, 1, W, true);
    	
    	Tensor diff = new Tensor(N, T, 1, W, true);
    	
//    	Tensor diff3 = new Tensor(N, T, 1, W, true);
    	
//    	kernel.backward(input, delta, diff, gamma, dgamma, dbeta);
//    	
//    	diff.showShape();
//    	diff.showDM();
//    	dgamma.showShape();
//    	dgamma.showDM();
//    	dbeta.showShape();
//    	dbeta.showDM();
    	
    	
    	

//    	kernel.backward3(input, delta, diff3, gamma, dgamma, dbeta);
//    	
//    	
//    	diff3.showDM();
    	
    }

}
