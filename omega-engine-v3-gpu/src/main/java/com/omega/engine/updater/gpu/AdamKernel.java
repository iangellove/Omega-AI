package com.omega.engine.updater.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.network.BPNetwork;
import com.omega.engine.nn.network.Network;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

public class AdamKernel {
	
	public Network net;
	
	public Tensor mean_w;
	
	public Tensor var_w;
	
	public Tensor mean_b;
	
	public Tensor var_b;
	
	private float beta1 = 0.9f;
	
	private float beta2 = 0.999f;
	
	private CUfunction function;
	
	private CUfunction bn_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private Pointer kernelParameters;
	
	private Pointer kernelBiasParameters;
	
	public AdamKernel(int weightLength,Network net) {
		this.mean_w = new Tensor(1, 1, 1, weightLength, true);
		this.var_w = new Tensor(1, 1, 1, weightLength, true);
		this.net = net;
		init();
	}
	
	public AdamKernel(int weightLength,int biasLength,Network net) {
		this.mean_w = new Tensor(1, 1, 1, weightLength, true);
		this.var_w = new Tensor(1, 1, 1, weightLength, true);
		this.mean_b = new Tensor(1, 1, 1, biasLength, true);
		this.var_b = new Tensor(1, 1, 1, biasLength, true);
		this.net = net;
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

				function = CUDAModules.getFunctionByModule("H://updater.cu", "adam");
				
			}
			
			if(bn_function == null) {
				
				bn_function = CUDAModules.getFunctionByModule("H://updater.cu", "adam_bn");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void updateW(Tensor diffW,Tensor weight,float lr) {
		
		try {

	        /**
	         * 设置入参
	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
	         */ 
			kernelParameters = Pointer.to(
					Pointer.to(diffW.getGpuData()),
	        		Pointer.to(weight.getGpuData()),
	                Pointer.to(mean_w.getGpuData()),
	                Pointer.to(var_w.getGpuData()),
	                Pointer.to(new float[]{beta1}),
	                Pointer.to(new float[]{beta2}),
	                Pointer.to(new float[]{lr}),
	                Pointer.to(new int[]{diffW.dataLength}),
	                Pointer.to(new int[]{net.number}),
	                Pointer.to(new int[]{net.train_time})
	            );

			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(diffW.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
			
//			System.out.println("diffW:"+net.train_time);
//			diffW.showDM();
			
//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void updateGama(Tensor diffW,Tensor weight,float lr) {
		
		try {

	        /**
	         * 设置入参
	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
	         */ 
			kernelParameters = Pointer.to(
					Pointer.to(diffW.getGpuData()),
	        		Pointer.to(weight.getGpuData()),
	                Pointer.to(mean_w.getGpuData()),
	                Pointer.to(var_w.getGpuData()),
	                Pointer.to(new float[]{beta1}),
	                Pointer.to(new float[]{beta2}),
	                Pointer.to(new float[]{lr}),
	                Pointer.to(new int[]{diffW.dataLength}),
	                Pointer.to(new int[]{net.train_time})
	            );

			cuLaunchKernel(bn_function,
		            this.CAFFE_GET_BLOCKS(diffW.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        );
			
//	        JCudaDriver.cuCtxSynchronize();
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void updateB(Tensor diffB,Tensor bias,float lr) {
		
		try {
			
	        /**
	         * 设置入参
	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
	         */ 
			kernelBiasParameters = Pointer.to(
					Pointer.to(diffB.getGpuData()),
	        		Pointer.to(bias.getGpuData()),
	                Pointer.to(mean_b.getGpuData()),
	                Pointer.to(var_b.getGpuData()),
	                Pointer.to(new float[]{beta1}),
	                Pointer.to(new float[]{beta2}),
	                Pointer.to(new float[]{lr}),
	                Pointer.to(new int[]{diffB.dataLength}),
	                Pointer.to(new int[]{net.number}),
	                Pointer.to(new int[]{net.train_time})
	            );
	        
			cuLaunchKernel(function,
		            this.CAFFE_GET_BLOCKS(diffB.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelBiasParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void updateBeta(Tensor diffB,Tensor bias,float lr) {
		
		try {
			
	        /**
	         * 设置入参
	         * float *diffW, float *weight,float *mw,float *vw,float beta1,float beta2,float learnRate, int n
	         */ 
			kernelBiasParameters = Pointer.to(
					Pointer.to(diffB.getGpuData()),
	        		Pointer.to(bias.getGpuData()),
	                Pointer.to(mean_b.getGpuData()),
	                Pointer.to(var_b.getGpuData()),
	                Pointer.to(new float[]{beta1}),
	                Pointer.to(new float[]{beta2}),
	                Pointer.to(new float[]{lr}),
	                Pointer.to(new int[]{diffB.dataLength}),
	                Pointer.to(new int[]{net.train_time})
	            );
	        
			cuLaunchKernel(bn_function,
		            this.CAFFE_GET_BLOCKS(diffB.dataLength),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelBiasParameters, null // Kernel- and extra parameters
		        );

//	        JCudaDriver.cuCtxSynchronize();

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
	    	
	    	float[] x1 = new float[] {0.1f,0.2f,0.3f,0.4f,-0.5f,0.6f,-0.7f,-0.8f,0.9f,0.01f,0.11f,-0.12f,0.13f,0.14f,0.15f,-0.16f};
	    	
	    	float[] bias1 = RandomUtils.order(N * C * H * W, 0.00001f, 0.00001f);
	    	
	    	Tensor w = new Tensor(N, C, H, W, x1, true);
	    	
	    	Tensor output = new Tensor(N, C, H, W, true);
	    	
	    	Tensor delta = new Tensor(N, C, H, W, bias1, true);
	    	
	    	Tensor diff = new Tensor(N, C, H, W, true);
	    	
	    	BPNetwork net = new BPNetwork(new SoftmaxWithCrossEntropyLoss());
	    	net.train_time = 1;
	    	net.number = N;
	    	AdamKernel k = new AdamKernel(bias1.length, net);

	    	k.updateGama(delta, w, 0.001f);
	    	
	    	w.showDM();
	    	

			CUDAMemoryManager.free();
			
	   }
	
	
	
}
