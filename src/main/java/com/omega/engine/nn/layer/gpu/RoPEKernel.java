package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Transformer;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

/**
 * Root Mean Sqrt Normalization
 * p = x^2
 * mean = 1/n∑x^2
 * rms = rsqrt(1/n∑x^2)
 * rms_norm = x * rms
 * drms = sum(x * diff)
 * dmean = -0.5 * (mean).pow(-1.5)
 * dp = 1/n
 * dx = rms * diff + sum(x * diff) * -0.5 * (mean).pow(-1.5) / n * 2 * x
 */
public class RoPEKernel extends BaseKernel{
	
	/**
	 * 向前方法
	 */
	private CUfunction forward_function;
	
	/**
	 * 反向传播方法
	 */
	private CUfunction backward_function;
	
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	/**
	 * 前向方法参数
	 */
	private Pointer forwardParameters;
	private Pointer backwardParameters;
	
	public RoPEKernel() {
		init();
	}
	
	public void initFunction() {
		
		try {
			
			if(forward_function == null) {
				forward_function = CUDAModules.getLocalFunctionByModule("RoPEKernel.cu", "rope_norm");
			}

			if(backward_function == null) {
				backward_function = CUDAModules.getLocalFunctionByModule("RoPEKernel.cu", "rope_norm");
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
	
	public void initBackward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(Tensor cos,Tensor sin, Tensor input, Tensor output) {
		
		try {
			
			int nrow = input.number * input.channel;
			
			int ncol = input.height * input.width;
			
			/**
			 * const float* x, float* dst,float* c_cos,float* c_sin, int ncols
			 */
			forwardParameters = Pointer.to(
					Pointer.to(input.getGpuData()),
					Pointer.to(output.getGpuData()),
					Pointer.to(cos.getGpuData()),
					Pointer.to(sin.getGpuData()),
					Pointer.to(new int[] {ncol})
	            );
			
			int[] block_dims = new int[] {1, 256, 1};
			
			int num_blocks_x = (ncol + 2*256 - 1) / (2*256);
			
			int[] block_nums = new int[] {nrow, num_blocks_x, 1};
			
			checkCUDA(cuLaunchKernel(forward_function,
					block_nums[0], block_nums[1], block_nums[2],      // Grid dimension
					block_dims[0], block_dims[1], block_dims[2],      // Block dimension
					0, null,               // Shared memory size and stream
					forwardParameters, null // Kernel- and extra parameters
				));
			
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff) {
		
//		try {
//			
//			dgamma.clearGPU();
//			
//			/**
//			 * float* __restrict__ out, float* __restrict__ dweight, float* __restrict__ smean, float* __restrict__ rms, const float*  __restrict__ inp, const float*  __restrict__ delta, const float* __restrict__ weight, int N, int C
//			 */
//			backwardParameters = Pointer.to(
//					Pointer.to(diff.getGpuData()),
//					Pointer.to(dgamma.getGpuData()),
//					Pointer.to(d_mean),
//					Pointer.to(d_rms),
//					Pointer.to(input.getGpuData()),
//					Pointer.to(delta.getGpuData()),
//					Pointer.to(gamma.getGpuData()),
//					Pointer.to(new int[] {B}),
//					Pointer.to(new int[] {W})
//	            );
//			
//			checkCUDA(cuLaunchKernel(backward_function,
//					B, 1, 1,      // Grid dimension
//					CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
//					0, null,               // Shared memory size and stream
//					backwardParameters, null // Kernel- and extra parameters
//				));
//
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//		}
		
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}

    public void showDM(String id,CUdeviceptr d,float[] data) {
    	JCudaDriver.cuMemcpyDtoH(Pointer.to(data), d, data.length * Sizeof.FLOAT);
    	System.out.println(id + ":"+JsonUtils.toJson(data));
    }
    
    public static void main(String[] args) {
    	
    	 try {

			CUDAModules.initContext();
			
			int N = 2;
	    	int T = 1;
	    	int W = 5;

	    	float[] data = RandomUtils.order(N * T * W, 0.1f, 0.1f);
	    	
	    	Tensor input = new Tensor(N * T, 1, 1, W, data, true);

	    	Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.one(N * T * W), true);
	    	
//    	    	Tensor delta = new Tensor(N * T, 1, 1, W, MatrixUtils.order(N * T * W, 0.1f, 0.1f), true);

	    	float[] gammaData = RandomUtils.order(W, 0.1f, 0.1f);
	    	
	    	Tensor gamma = new Tensor(1, 1, 1, W, gammaData, true);
	    	
	    	Tensor dgamma = new Tensor(1, 1, 1, W, true);
	    	
	    	Transformer tf = new Transformer();
//    			tf.number = N * T;
	    	
	    	RMSLayer rms = new RMSLayer(tf);
	    	rms.gamma = gamma;
	    	rms.diffGamma = dgamma;
	    	input.showDM();
	    	for(int i = 0;i<10;i++) {
	    		rms.forward(input);
	    		rms.getOutput().showDM();
	    		rms.back(delta);
	    		rms.diff.showDM();
	    		rms.diffGamma.showDM();
	    	}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}

    }

}
