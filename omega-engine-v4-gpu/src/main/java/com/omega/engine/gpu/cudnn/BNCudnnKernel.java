package com.omega.engine.gpu.cudnn;

import static jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
import static jcuda.jcudnn.cudnnTensorFormat.CUDNN_TENSOR_NCHW;

import com.omega.common.data.Tensor;
import com.omega.common.utils.CheckArrayUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.network.RunModel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnBatchNormMode;
import jcuda.jcudnn.cudnnTensorDescriptor;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

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
	private double momentum = 0.1f;
    
	private Pointer alpha_P = Pointer.to(new float[] { 1 });
	private Pointer beta_P = Pointer.to(new float[] { 0 });
	
	private Tensor mean;
	private Tensor var;
    
	private Tensor runingMean;
	private Tensor runingVar;
	
	private cudnnTensorDescriptor normTensorDesc;
	private cudnnTensorDescriptor dstTensorDesc;
	
	private Pointer diff;
	
	
	public BNCudnnKernel(BNType bnType,int C,int H,int W) {
		this.bnType = bnType;
		this.C = C;
		this.H = H;
		this.W = W;
		init();
	}
	
	public void init() {
		
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
	
