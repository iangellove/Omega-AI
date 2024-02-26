package com.omega.engine.gpu.cudnn;

import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcudnn.JCudnn;
import jcuda.jcudnn.cudnnAttnDescriptor;
import jcuda.jcudnn.cudnnDataType;
import jcuda.jcudnn.cudnnDropoutDescriptor;
import jcuda.jcudnn.cudnnMathType;
import jcuda.jcudnn.cudnnSeqDataDescriptor;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

/**
 * MultiHeadAttentionCudnnKernel
 * @author Administrator
 *
 */
public class MultiHeadAttentionCudnnKernel extends BaseKernel{
	
	public int attnMode = JCudnn.CUDNN_ATTN_DISABLE_PROJ_BIASES;
	
	private int headNum = 1;
	
	private int beamSize = 1;
	
	private double smScaler = 1.0f;

	public int N;
	
	public int qSize;  //Q输入维度
	public int kSize;  //K输入维度
	public int vSize;  //V输入维度
	
	public int qProjSize;  //Q每头输出维度 hiddenSize
	public int kProjSize;  //K每头输出维度 hiddenSize
	public int vProjSize;  //V每头输出维度 hiddenSize
	public int oProjSize;  //O多头总输出维度 hiddenSize * headNum
	
	private int seqLenQ;  //最大QO序列数
	private int seqLenK;  //最大KV序列数
	
	private boolean resLink = false;
	
	private Pointer workspace;
	private Pointer reserveSpace;
	
	private cudnnAttnDescriptor attnDesc;
	
	private cudnnDropoutDescriptor dropoutDesc;
	
	private cudnnSeqDataDescriptor qDesc;
	private cudnnSeqDataDescriptor kDesc;
	private cudnnSeqDataDescriptor vDesc;
	private cudnnSeqDataDescriptor oDesc;
	
	private int dataType = cudnnDataType.CUDNN_DATA_FLOAT;
	private int compPrec = cudnnDataType.CUDNN_DATA_FLOAT;
	
	private int[] qSeqArray;
	private int[] kSeqArray;
	
	private int[] loWinIdx;
	private int[] hiWinIdx;
	
	private float dropout = 0;
	
	private long workSize = 0;
	
	private long reserveSize = 0;
	
	private long[] weightSpaceSize = { 0 };

	public MultiHeadAttentionCudnnKernel(int time,int layerNum,int inputSize,int hiddenSize,boolean bidirectional,int rnnMode,float dropout,boolean hasBias) {

		this.dropout = dropout;
		
		init();
	}
	
	public void init() {
		
		attnDesc = new cudnnAttnDescriptor();

		qDesc = new cudnnSeqDataDescriptor();
		kDesc = new cudnnSeqDataDescriptor();
		vDesc = new cudnnSeqDataDescriptor();
		oDesc = new cudnnSeqDataDescriptor();

        dropoutDesc = new cudnnDropoutDescriptor();
        
        workspace = new Pointer();
        reserveSpace = new Pointer();
	}
	
	public void init(int number,int time) {
		
		if(this.N != number) {
			
			this.N = number;
			
	        long seed = 1337; // Pick a seed.
	        
	        JCudnn.cudnnCreateDropoutDescriptor(dropoutDesc);
	        
	        long stateSizeArray[] = { 0 };
	        Pointer states = new Pointer();
	        JCudnn.cudnnDropoutGetStatesSize(CudnnHandleManager.getHandle(), stateSizeArray);
	        long stateSize = stateSizeArray[0];

	        JCuda.cudaMalloc(states, stateSize);

	        handle(JCudnn.cudnnSetDropoutDescriptor(dropoutDesc,
	        	CudnnHandleManager.getHandle(),
	            dropout,
	            states,
	            stateSize,
	            seed));
	        
	        handle(JCudnn.cudnnSetAttnDescriptor(attnDesc,
	        		attnMode,
	        		headNum,
	        		smScaler,
	        		dataType,
	        		compPrec,
	        		cudnnMathType.CUDNN_DEFAULT_MATH,
	        		null,  //dropoutRate > 0.0 ? drop_desc : NULL,
	        		null,
	        		qSize,
	        		kSize,
	        		vSize,
	        		qProjSize,
	        		kProjSize,
	        		vProjSize,
	        		oProjSize,
	        		seqLenQ,
	        		seqLenK,
	        		N,
	        		beamSize));
	        
	        long[] sizeWeights = {0}, sizeWkspace = {0}, sizeReserve = {0};
	        
	        handle(JCudnn.cudnnGetMultiHeadAttnBuffers(CudnnHandleManager.getHandle(),
	        		attnDesc,
	        		sizeWeights,
	        		sizeWkspace,
	        		sizeReserve));
	        
	        
	        JCuda.cudaDeviceSynchronize();

		}
		
	}
	
//	public long weightSize() {
//		JCudnn.cudnnGetRNNWeightSpaceSize(CudnnHandleManager.getHandle(), rnnDesc, getWeightSpaceSize());
//        long weightsSize = getWeightSpaceSize()[0];
////        System.out.println(weightsSize / Sizeof.FLOAT);
//        return weightsSize;
//	}
//	
//	public void initWeights(Tensor w) {
//		
////		float[] data = new float[] {0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.90000004f,1.0f,1.1f,1.2f,1.3000001f,1.4f,
////				0.0f,0.2f,0.4f,0.6f,0.8f,1.0f,1.2f,1.4f,1.6f,1.8000001f,2.0f,2.2f,2.4f,2.6000001f,2.8f,3.0f,3.2f,3.4f,3.6000001f,3.8f,4.0f,4.2000003f,4.4f,4.6f,4.8f,
////				0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.2f,0.2f,0.2f,0.2f};
//		
////		float[] data = new float[] {0.0f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.90000004f,1.0f,1.1f,1.2f,1.3000001f,1.4f,
////				0.0f,0.2f,0.4f,0.6f,0.8f,1.0f,1.2f,1.4f,1.6f,1.8000001f,2.0f,2.2f,2.4f,2.6000001f,2.8f,3.0f,3.2f,3.4f,3.6000001f,3.8f,4.0f,4.2000003f,4.4f,4.6f,4.8f};
//		
////		float[] data = new float[] {0.0f,0.5f,1.0f,0.1f,0.6f,1.1f,0.2f,0.7f,1.2f,0.3f,0.8f,1.3000001f,0.4f,0.90000004f,1.4f,
////				0.0f,1.0f,2.0f,3.0f,4.0f,0.2f,1.2f,2.2f,3.2f,4.2000003f,0.4f,1.4f,2.4f,3.4f,4.4f,0.6f,1.6f,2.6000001f,3.6000001f,4.6f,0.8f,1.8000001f,2.8f,3.8f,4.8f,
////				0.1f,0.1f,0.1f,0.1f,0.1f,0.2f,0.2f,0.2f,0.2f,0.2f};
////		
////		w.data = data;
//		
////		w.fill(0.1f);
////		float[] w1 = new float[] {0.0f,1.0f,2.0f,3.0f,4.0f,0.2f,1.2f,2.2f,3.2f,4.2000003f,0.4f,1.4f,2.4f,3.4f,4.4f,0.6f,1.6f,2.6000001f,3.6000001f,4.6f,0.8f,1.8000001f,2.8f,3.8f,4.8f};
////		System.out.println(JsonUtils.toJson(MatrixUtils.transpose(w1, 5, 5)));
//		
////		System.out.println(w.dataLength);
////		
////		w.hostToDevice();
////		
////		System.out.println(JsonUtils.toJson(w.syncHost()));
//		
//		float stddev = (float) Math.sqrt(2.0d / (inputSize + hiddenSize)); // glorot_uniform like tensorflow    
//		 
//		curandGenerator generator = new curandGenerator();
//		 
//		JCurand.curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
//		JCurand.curandSetPseudoRandomGeneratorSeed(generator, 1337);
//		 
//		JCurand.curandGenerateNormal(generator, w.getGpuData(), w.getDataLength(), 0, stddev);
//	}
//	
//	public void forward(RunModel RUN_MODEL,Tensor input, Tensor hx, Tensor cx, Tensor weight, Tensor output, Tensor hy, Tensor cy) {
//		// TODO Auto-generated method stub
//
//		if(RUN_MODEL == RunModel.TRAIN) {
////			input.showDM();
//			//cudnnHandle handle, cudnnRNNDescriptor rnnDesc, int fwdMode, Pointer devSeqLengths, cudnnRNNDataDescriptor xDesc, Pointer x,
//			//cudnnRNNDataDescriptor yDesc, Pointer y, cudnnTensorDescriptor hDesc, Pointer hx, Pointer hy, cudnnTensorDescriptor cDesc, Pointer cx,
//			//Pointer cy, long weightSpaceSize, Pointer weightSpace, long workSpaceSize, Pointer workSpace, long reserveSpaceSize, Pointer reserveSpace
//			
//			JCudnn.cudnnMultiHeadAttnForward(CudnnHandleManager.getHandle(), attnDesc, -1, loWinIdx, hiWinIdx, devSeqLengthsQO, devSeqLengthsKV,
//					qDesc, queries, residuals, kDesc, keys, vDesc, values, oDesc, out, weightSizeInBytes, weights,
//					workSpaceSizeInBytes, workSpace, reserveSpaceSizeInBytes, reserveSpace)
//			
//			handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(),
//					rnnDesc,
//					cudnnForwardMode.CUDNN_FWD_MODE_TRAINING,
//					seqP,
//					xDesc,
//					input.getGpuData(),
//					yDesc,
//					output.getGpuData(),
//					hDesc,
//					hx.getGpuData(), //hx
//					hy.getGpuData(), //hy
//					cDesc,
//					cx.getGpuData(), //cx
//					cy.getGpuData(), //cy
//					getWeightSpaceSize()[0],
//					weight.getGpuData(),
//					workSize,
//					workspace,
//					reserveSize,
//					reserveSpace));
//		
//		}else {
//			
//			handle(JCudnn.cudnnRNNForward(CudnnHandleManager.getHandle(),
//					rnnDesc,
//					cudnnForwardMode.CUDNN_FWD_MODE_INFERENCE,
//					seqP,
//					xDesc,
//					input.getGpuData(),
//					yDesc,
//					output.getGpuData(),
//					hDesc,
//					hx.getGpuData(), //hx
//					hy.getGpuData(), //hy
//					cDesc,
//					cx.getGpuData(), //cx
//					cy.getGpuData(), //cy
//					getWeightSpaceSize()[0],
//					weight.getGpuData(),
//					workSize,
//					workspace,
//					reserveSize,
//					reserveSpace));
//		}
//		
//	}
//	
//	public void dw(Tensor delta, Tensor output, Tensor input, Tensor hx, Tensor dw) {
//		// TODO Auto-generated method stub
//		
//		// cudnnRNNBackwardWeights adds to the data in dw.
//		dw.clearGPU();
//		
//		JCudnn.cudnnRNNBackwardWeights_v8(CudnnHandleManager.getHandle(),
//				rnnDesc,
//				CUDNN_WGRAD_MODE_ADD,
//				seqP,
//				xDesc,
//				input.getGpuData(),
//				hDesc,
//				hx.getGpuData(), //hx
//				yDesc,
//				output.getGpuData(),
//				dw.getDataLength() * Sizeof.FLOAT,
//				dw.getGpuData(),
//				workSize,
//				workspace,
//				reserveSize,
//				reserveSpace);
//
//	}
//
//	public void dx(Tensor delta,Tensor dhy,Tensor dcy, Tensor output, Tensor hx, Tensor cx, Tensor weight, Tensor diff, Tensor dhx,Tensor dcx) {
//		// TODO Auto-generated method stub
//		
//		Pointer dhy_p = null;
//		
//		if(dhy != null) {
//			dhy_p = dhy.getGpuData();
//		}
//
//		handle(JCudnn.cudnnRNNBackwardData_v8(CudnnHandleManager.getHandle(),
//				rnnDesc,
//				seqP,
//				yDesc,
//				output.getGpuData(),
//				delta.getGpuData(),
//				xDesc,
//				diff.getGpuData(),
//				hDesc,
//				hx.getGpuData(), //hx
//				dhy_p,  //dhy
//				dhx.getGpuData(),  //dhx
//				cDesc,
//				cx.getGpuData(), //cx
//				dcy.getGpuData(),  //dcy
//				dcx.getGpuData(),  //dcx
//				weight.getDataLength() * Sizeof.FLOAT,
//				weight.getGpuData(),
//				workSize,
//				workspace,
//				reserveSize,
//				reserveSpace));
//		
//	}
	
	/**
	 * Handle.
	 *
	 * @param returnCode the return run
	 */
	public static void handle(final int returnCode) {
		if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
		      System.err.println(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
		      throw new RuntimeException(jcuda.jcudnn.cudnnStatus.stringFor(returnCode));
		}
	}
	
	public static String checkError(final int returnCode) {
	    if (returnCode != jcuda.jcudnn.cudnnStatus.CUDNN_STATUS_SUCCESS) {
	        return jcuda.jcudnn.cudnnStatus.stringFor(returnCode);
	    }else {
	    	return "success";
	    }
	}

	public long[] getWeightSpaceSize() {
		return weightSpaceSize;
	}
	
	private static void initGPUData(Pointer data, int numElements, float a, float b){
        // Note: The original sample used a kernel to initialize the memory.
        // Using a host array to fill the memory is less efficient, but does
        // not require any custom kernels, and is done here for brevity.
        float array[] = RandomUtils.order(numElements, a, b);
        JCuda.cudaMemcpy(data, Pointer.to(array), 
            numElements * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
    }

}
