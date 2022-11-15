package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class SoftmaxKernel{
	
	private int N = 0;
	
	private CUfunction softmax_function;
	
	private CUfunction softmax_backward_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer kernelParameters;
	
	private Pointer backKernelParameters;
	
	public SoftmaxKernel() {
		init();
	}
	
	public void initFunction() {
		
		try {

			if(softmax_function == null) {
				
				softmax_function = CUDAModules.getFunctionByModule("H://SoftmaxKernel.cu", "softmax");
        
			}
			
			if(softmax_backward_function == null) {
				
				softmax_backward_function = CUDAModules.getFunctionByModule("H://SoftmaxKernel.cu", "softmax_back");
        
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
	
	public void forward(Tensor input,Tensor output) {
		
//		if(kernelParameters == null || this.N != output.number) {
			/**
			 * float *input, float *output, int batch, int n, float temp
			 */
			kernelParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(new int[] {input.number}),
	                Pointer.to(new int[] {input.width})
	            );
			
			this.N = output.number;
			
//		}
		
		cuLaunchKernel(softmax_function,
				input.number,  1, 1,      // Grid dimension
	            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
		
//		JCudaDriver.cuCtxSynchronize();
		
	}
	
	public void backward(Tensor output,Tensor currentLabel,Tensor diff) {

//		if(backKernelParameters == null) {

			/**
			 * float* x,float* mean,float* var,int number,int channel,int height,int width
			 */
			backKernelParameters = Pointer.to(
	                Pointer.to(output.getGpuData()),
	                Pointer.to(currentLabel.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {diff.dataLength})
	            );

//		}
		
		cuLaunchKernel(softmax_backward_function,
				this.CAFFE_GET_BLOCKS(diff.dataLength),  1, 1,      // Grid dimension
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
	
	public void cpuForward(Tensor input,Tensor output) {
		float[] dest = new float[input.channel * input.height * input.width];
		
		for(int n = 0;n<input.number;n++) {
			input.copy(n, dest);
			float max = MatrixOperation.max(dest);
			float[] temp = MatrixOperation.subtraction(dest, max);
			temp = MatrixOperation.exp(temp);
			float sum = MatrixOperation.sum(temp);
			for(int i = 0;i<temp.length;i++) {
				output.data[n * output.channel * output.height * output.width + i] = temp[i] / sum;
			}
		}
		
//		System.out.println(JsonUtils.toJson(output.data));
		
	}
	
	public void cpuForward2(Tensor input,Tensor output){
		
		for(int id = 0;id<input.number;id++) {
			float max = -3.402823466e+38F;
			float sum = 0;
			for(int i = 0;i<input.width;i++) {
				if(max <= input.data[id * input.width + i]) {
					max = input.data[id * input.width + i];
				}
			}
			for(int i = 0;i<input.width;i++){
		        float e = (float) Math.exp(input.data[id * input.width + i] - max);
		        sum += e;
		        output.data[id * input.width + i] = e;
		    }
			for(int i = 0;i<input.width;i++){
		        output.data[id * input.width + i] /= sum;
		    }
		}
	}
	
	public void cpuBackward(Tensor output,Tensor currentLabel,Tensor diff) {
		// TODO Auto-generated method stub
		
		for(int i = 0;i<output.getDataLength();i++) {
			diff.data[i] = output.data[i] - currentLabel.data[i];
		}
		
//		System.out.println(JsonUtils.toJson(diff.data));

	}
	
	public static void main(String[] args) {
		
		int N = 2;
		int C = 1;
		int H = 1;
		int W = 10;
		
		float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
		
		Tensor input = new Tensor(N, C, H, W, x, true);
		
		Tensor output = new Tensor(N, C, H, W, true);
		
		Tensor output2 = new Tensor(N, C, H, W);
		
		SoftmaxKernel k = new SoftmaxKernel();
		
		k.forward(input, output);
		
		output.showDM();
		
		k.cpuForward2(input, output2);
		
		System.out.println(JsonUtils.toJson(output2.data));
		
	}
	
}
