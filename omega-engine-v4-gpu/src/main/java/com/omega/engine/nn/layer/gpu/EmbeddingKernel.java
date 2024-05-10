package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.cudaError;

public class EmbeddingKernel extends BaseKernel{

	private CUfunction function;
	
	private CUfunction back_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer kernelParameters;
	
	private Pointer kernelBackParameters;
	
	public EmbeddingKernel() {
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

				function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"EmbeddingKernel.cu", "EmbeddingFW");
				
			}
			
			if(back_function == null) {

				back_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"EmbeddingKernel.cu", "EmbeddingGrad");
				
			}
			
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
	
	public void forward(Tensor input,Tensor weight,Tensor output) {
		
		try {
			
//			if(kernelParameters == null || input.number != this.N){

		        /**
		         * 设置入参
		         *  float *output,
                    const float *table,
                    const float *ids,
                    const int N,
                    const int K,
                    const int D
		         */ 
		        kernelParameters = Pointer.to(
		        		Pointer.to(output.getGpuData()),
		        		Pointer.to(weight.getGpuData()),
		        		Pointer.to(input.getGpuData()),
		        		Pointer.to(new int[]{weight.height}),
		        		Pointer.to(new int[]{input.getDataLength()}),
		        		Pointer.to(new int[]{weight.width})
		            );
		        
		        this.N = input.number;
		        
//			}
			
			int gridx = 2 * CUDAModules.props.multiProcessorCount;
		    int[] threads = new int[] {256, 4, 1};
		    int[] grids = new int[] {gridx, 1, 1};
			
		    checkCUDA(cuLaunchKernel(function,
					grids[0],  grids[1], grids[2],      // Grid dimension
					threads[0], threads[1], threads[2],
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

//	        JCudaDriver.cuCtxSynchronize();
//	        output.syncHost();
//	        output.showDMByNumber(0);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor delta,Tensor dw,Tensor input) {
		
		try {
			
//			dw.valueGPU(0);
			
			if(kernelBackParameters == null || delta.number != this.N){

		        /**
		         * 设置入参
		         * float* table,
                   const float* output,
                   const float* ids,
                   const int N,
                   const int K,
                   const int D
		         */ 
				kernelBackParameters = Pointer.to(
		        		Pointer.to(dw.getGpuData()),
		        		Pointer.to(delta.getGpuData()),
		                Pointer.to(input.getGpuData()),
		                Pointer.to(new int[]{dw.height}),
		        		Pointer.to(new int[]{input.dataLength}),
		        		Pointer.to(new int[]{dw.width})
		            );
		        
		        this.N = delta.number;
		        
			}
			
			int gridx = 2 * CUDAModules.props.multiProcessorCount;
		    int[] threads = new int[] {128, 8, 1};
		    int[] grids = new int[] {gridx, 1, 1};

		    checkCUDA(cuLaunchKernel(back_function,
					grids[0],  grids[1], grids[2],      // Grid dimension
					threads[0], threads[1], threads[2],
		            0, null,               // Shared memory size and stream
		            kernelBackParameters, null // Kernel- and extra parameters
		        ));
//			delta.showDMByNumber(0);
//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public static void main(String[] args) {
		
		int N = 2;
		int W = 3;
		
		int OW = 5;
		
		float[] data = new float[] {2.0f, 0.0f};
    	
    	Tensor input = new Tensor(N, 1, 1, 1, data, true);
    	
    	Tensor output = new Tensor(N, 1, 1, OW, true);
    	
    	float[] wdata = RandomUtils.order(W * OW, 0.1f, 0.1f);
    	
    	Tensor weight = new Tensor(1, 1, W, OW, wdata, true);

    	Tensor delta = new Tensor(N, 1, 1, OW, MatrixUtils.order(N * OW, 0.1f, 0.1f), true);
    	
    	Tensor dw = new Tensor(1, 1, W, OW, true);
    	
    	EmbeddingKernel kernel = new EmbeddingKernel();
    	
    	kernel.forward(input, weight, output);
    	
//    	input.showDM();
//    	weight.showDM();
    	output.showDM();
    	
    	kernel.backward(delta, dw, input);
		
//    	dw.showDM();
	}
	
}
