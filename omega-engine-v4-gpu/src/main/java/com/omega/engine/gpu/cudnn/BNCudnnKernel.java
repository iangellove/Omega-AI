package com.omega.engine.gpu.cudnn;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.network.RunModel;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnTensorDescriptor;

/**
 * BNCudnnKernel
 * @author Administrator
 *
 */
public class BNCudnnKernel extends BNBaseKernel{
	
	private BNType bnType;

	private int mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
	
	private int C;
	private int H;
	private int W;
	
	private double eps = 1e-5;
	private double momentum = 0.01f;
    
	private Pointer alpha_P = Pointer.to(new float[] { 1 });
	private Pointer beta_P = Pointer.to(new float[] { 0 });
	
	private Tensor mean;
	private Tensor var;
    
	private Tensor runingMean;
	private Tensor runingVar;
	
	private cudnnTensorDescriptor normTensorDesc;
	private cudnnTensorDescriptor dstTensorDesc;
	
//	private Pointer diff;
	
	private CUfunction normalize_test_function;
	
	private Pointer normalize_test_Parameters;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	
	public BNCudnnKernel(BNType bnType,int C,int H,int W) {
		this.bnType = bnType;
		this.C = C;
		this.H = H;
		this.W = W;
		init();
	}
	
	public void init() {
		
//		if(normalize_test_function == null) {
//			normalize_test_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "normalize_test_kernel");
//		}
		
		if(bnType == BNType.fully_bn) {
			
			mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
			
			mean = new Tensor(1, 1, 1, W, true);
			var = new Tensor(1, 1, 1, W, true);
		    
		    runingMean = new Tensor(1, 1, 1, W, true);
		    runingVar = new Tensor(1, 1, 1, W, true);
		}else {
			
			mode = cudnnBatchNormMode.CUDNN_BATCHNORM_SPATIAL;
			
			mean = new Tensor(1, 1, 1, C, true);
			var = new Tensor(1, 1, 1, C, true);
		    
		    runingMean = new Tensor(1, 1, 1, C, true);
		    runingVar = new Tensor(1, 1, 1, C, true); 
		}
		
		normTensorDesc = new cudnnTensorDescriptor();
		dstTensorDesc = new cudnnTensorDescriptor();
	   
	    JCudnn.cudnnCreateTensorDescriptor(normTensorDesc);
	    JCudnn.cudnnCreateTensorDescriptor(dstTensorDesc);
	    
	}
	
	public void initForward(Tensor input) {
		
		if(input.number != this.N) {
			
			this.N = input.number;
			
			if(bnType == BNType.fully_bn) {

				JCudnn.cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, W, 1, 1);
				
			    JCudnn.cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, W, 1, 1);
			}else {

				JCudnn.cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
			    
			    JCudnn.cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1);
			}
			
		}
		
	}
	
	public void forward(RunModel RUN_MODEL, Tensor gamma, Tensor beta, Tensor input, Tensor output) {
			
		initForward(input);
		
		if(RUN_MODEL == RunModel.TRAIN) {
			
			CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardTraining(CudnnHandleManager.getHandle(), mode,
			    		alpha_P, beta_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, output.getGpuData(),
			    		normTensorDesc, gamma.getGpuData(), beta.getGpuData(), momentum, runingMean.getGpuData(), runingVar.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));

		}else {
			
			CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationForwardInference(CudnnHandleManager.getHandle(), mode,
		    		alpha_P, beta_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, output.getGpuData(), normTensorDesc, gamma.getGpuData(), beta.getGpuData(),
		    		runingMean.getGpuData(), runingVar.getGpuData(), eps));
			
//			normalize_test(input, gamma, beta, output);
			
		}

	}
	
	public void normalize_test(Tensor input,Tensor gama, Tensor beta, Tensor output) {
		
		try {
			
			/**
			 * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial
			 */
			normalize_test_Parameters = Pointer.to(
					Pointer.to(new int[] {N * C * H * W}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(runingMean.getGpuData()),
	                Pointer.to(runingVar.getGpuData()),
	                Pointer.to(gama.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {H * W}),
	                Pointer.to(new float[] {(float) eps})
	            );
			
	        cuLaunchKernel(normalize_test_function,
		            this.CAFFE_GET_BLOCKS(N * C * H * W),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            normalize_test_Parameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
//		System.out.println(delta.channel+":"+C);
//		
//		if(this.diff == null) {
//			this.diff = CUDAMemoryManager.getPointer(diff.getDataLength());
//		}
//		
//		CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode,
//	    		alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, delta.getGpuData(), dstTensorDesc, this.diff,
//	    		normTensorDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
		
		CudnnHandleManager.handle(JCudnn.cudnnBatchNormalizationBackward(CudnnHandleManager.getHandle(), mode,
	    		alpha_P, beta_P, alpha_P, alpha_P, dstTensorDesc, input.getGpuData(), dstTensorDesc, delta.getGpuData(), dstTensorDesc, diff.getGpuData(),
	    		normTensorDesc, gamma.getGpuData(), dgamma.getGpuData(), dbeta.getGpuData(), eps, mean.getGpuData(), var.getGpuData()));
		
//		float[] test = new float[diff.getDataLength()];
//		
//		JCuda.cudaMemcpy(Pointer.to(test), this.diff, test.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
//		
//		System.out.println(diff.getDataLength()+":"+test.length);
//		
//		System.out.println(CheckArrayUtils.check(test, diff.syncHost()));
//		
//		copy_gpu(this.diff, diff.getGpuData(), diff.getDataLength(), 1, 1);
		
	}	
		
}
	
