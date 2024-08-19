package com.omega.engine.nn.layer.normalization.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Transformer;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

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
public class GNKernel extends BaseKernel{
	
	public BNType bnType = null;
	
	private int B;
	
	private int C;
	
	private int G;
	
	/**
	 * 向前方法
	 */
	private CUfunction forward_function;
	
	/**
	 * 反向传播方法
	 */
	private CUfunction backward_function;
	
	/**
	 * 前向方法参数
	 */
	private Pointer forwardParameters;
	/**
	 * 反向方法参数
	 */
	private Pointer backwardParameters;
	
	private CUdeviceptr d_mean;
	private CUdeviceptr d_var;
	
	public GNKernel(int G,BNType bnType) {
		this.bnType = bnType;
		this.G = G;
		init();
	}
	
	private void initKernel() {
		/**
		 * 申请向前传播参数显存
		 */
		if(this.d_mean != null) {
			CUDAMemoryManager.free(this.d_mean);
			CUDAMemoryManager.free(this.d_var);
		}

		this.d_mean = CUDAMemoryManager.getDevice(B * G);
		this.d_var = CUDAMemoryManager.getDevice(B * G);
	}
	
	public void initFunction() {
		
		try {
			
			if(forward_function == null) {
				forward_function = CUDAModules.getLocalFunctionByModule("GNKernel.cu", "groupnorm_forward_kernel");
			}

			if(backward_function == null) {
				backward_function = CUDAModules.getLocalFunctionByModule("GNKernel.cu", "groupnorm_backward_kernel");
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
		int batchSize = input.number;
		if(B != batchSize){
			this.B = batchSize;
			return false;
		}
		return true;
	}
	
	public void initBackward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
	}
	
	public void forward(Tensor gamma, Tensor beta, Tensor input, Tensor output) {

		try {
			
			boolean check = checkBatch(input);

			int img_size = input.height * input.width;
			int group_size = input.channel / G;
			int n_blocks  = input.number * G;
			int block_size = Math.max(Math.min(512, img_size * group_size), 32);

			if(!check) {

				initKernel();
				
				/**
				 *  const float* x, const float* weight, const float* bias,
    				float* out, float* mean, float* rstd,
    				int B, int C, int img_size, int group_size, int n_groups
				 */
				forwardParameters = Pointer.to(
						Pointer.to(input.getGpuData()),
		                Pointer.to(gamma.getGpuData()),
		                Pointer.to(beta.getGpuData()),
		                Pointer.to(output.getGpuData()),
						Pointer.to(d_mean),
						Pointer.to(d_var),
						Pointer.to(new int[] {input.number}),
						Pointer.to(new int[] {input.channel}),
						Pointer.to(new int[] {img_size}),
						Pointer.to(new int[] {group_size}),
						Pointer.to(new int[] {G})
		            );

			}
			
			cuLaunchKernel(forward_function,
					n_blocks, 1, 1,      // Grid dimension
					block_size, 1, 1,      // Block dimension
	        		0, null,               // Shared memory size and stream
		            forwardParameters, null // Kernel- and extra parameters
				);
			
//			showGPU(d_mean, B * G);
//			
//			showGPU(d_var, B * G);

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {
		
		try {
			
			int img_size = input.height * input.width;
			int group_size = input.channel / G;
			int n_blocks  = input.number * G;
			int block_size = Math.max(Math.min(512, img_size * group_size), 32 * group_size);
		
			/**
			 *   const float* dout, const float* x, const float* mean, const float* rstd, const float* weight,
    			 float* dx, float* dweight, float* dbias,
    			 int B, int C, int img_size, int group_size, int n_groups
			 */
			backwardParameters = Pointer.to(
					Pointer.to(delta.getGpuData()),
					Pointer.to(input.getGpuData()),
					Pointer.to(d_mean),
					Pointer.to(d_var),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(dgamma.getGpuData()),
	                Pointer.to(dbeta.getGpuData()),
					Pointer.to(new int[] {input.number}),
					Pointer.to(new int[] {input.channel}),
					Pointer.to(new int[] {img_size}),
					Pointer.to(new int[] {group_size}),
					Pointer.to(new int[] {G})
	            );

			cuLaunchKernel(backward_function,
					n_blocks, 1, 1,      // Grid dimension
					block_size, 1, 1,      // Block dimension
	        		0, null,               // Shared memory size and stream
		            backwardParameters, null // Kernel- and extra parameters
				);

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void showGPU(CUdeviceptr p,int len) {
		Tensor o = new Tensor(1,1,1,len,true);
		o.setGpuData(p);
		o.showDM();
	}
	
	public static void forwardCPU(int G,Tensor x,Tensor gamma,Tensor beta,Tensor output) {
		float[] mean = new float[x.number * G];
		float[] var = new float[x.number * G];
		int groupSize = x.channel / G;
		int imgSize = x.height * x.width;
		int once = groupSize * imgSize;
		for(int b = 0;b<x.number;b++) {
			for(int g = 0;g<G;g++) {
				float sum = 0.0f;
				float sum2 = 0.0f;
				for(int gs = 0;gs<once;gs++) {
					float val = x.data[b * G * once + g * once + gs];
					sum += val;
					sum2 += val * val;
				}
				float mean_val = sum / once;
				mean[b * G + g] = mean_val;
				var[b * G + g] = sum2 / once - mean_val * mean_val;
				for(int gs = 0;gs<groupSize * imgSize;gs++) {
					output.data[b * G * once + g * once + gs] = (float) ((x.data[b * G * once + g * once + gs] - mean_val) / Math.sqrt(var[b * G + g] + 1e-5f));
				}
			}
			for(int c = 0;c<x.channel;c++) {
				for(int i = 0;i<imgSize;i++) {
					float x_norm = output.data[b * x.channel * imgSize + c * imgSize + i];
					output.data[b * x.channel * imgSize + c * imgSize + i] = gamma.data[c] * x_norm + beta.data[c];
				}
			}
		}
		
		output.hostToDevice();
		output.showDM();
		
	}
	
	public static void main(String[] args) {
    	
   	  try {

			CUDAModules.initContext();
			
			int N = 2;
	    	int C = 64;
	    	int H = 4;
	    	int W = 4;
	    	int G = 32;

	    	float[] data = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
	    	
	    	Tensor input = new Tensor(N , C, H, W, data, true);

	    	Tensor delta = new Tensor(N , C, H, W, MatrixUtils.one(N * C * H * W), true);

	    	float[] gammaData = RandomUtils.val(C, 1.0f);
	    	
	    	float[] betaData = RandomUtils.val(C, 0.0f);
	    	
	    	Tensor gamma = new Tensor(1, 1, 1, C, gammaData, true);
	    	
	    	Tensor dgamma = new Tensor(1, 1, 1, C, true);
	    	
	    	Tensor beta = new Tensor(1, 1, 1, C, betaData, true);
	    	
	    	Tensor dbeta = new Tensor(1, 1, 1, C, true);

	    	Transformer tf = new Transformer();
	    	
	    	GNLayer rms = new GNLayer(G, tf);
	    	rms.bnType = BNType.conv_bn;
	    	rms.gamma = gamma;
	    	rms.diffGamma = dgamma;
	    	rms.beta = beta;
	    	rms.diffBeta = dbeta;
	    	input.showDM();
	    	for(int i = 0;i<1;i++) {
	    		rms.forward(input);
	    		rms.getOutput().showDM();
	    		rms.back(delta);
	    		rms.diff.showDM();
	    		rms.diffGamma.showDM();
	    		rms.diffBeta.showDM();
	    	}
			
	    	Tensor output = new Tensor(N, C, H, W, true);
	    	
	    	forwardCPU(G, input, gamma, beta, output);
	    	
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}

   }
	
}
