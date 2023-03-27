package com.omega.engine.nn.layer.normalization.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.gpu.BNBaseKernel;
import com.omega.engine.nn.network.RunModel;

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
public class BNKernel3 extends BNBaseKernel{
	
	private int C;
	private int H;
	private int W;
	
	private int spatial;
	
	private CUfunction mwa_function;
	
	/**
	 * 向前方法
	 */
	private CUfunction fast_mean_function;
	private CUfunction fast_var_function;
	private CUfunction normalize_function;
	private CUfunction normalize_test_function;
	
	/**
	 * 反向传播方法
	 */
	private CUfunction dgamma_function;
	private CUfunction dbeta_function;
	private CUfunction dbeta_full_function;
	private CUfunction dxhat_function;
	private CUfunction fast_dmean_function;
	private CUfunction fast_dvar_function;
	private CUfunction dx_function;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private float eta = 1e-5f;
	
	private float momentum = 0.01f;
	
	/**
	 * 前向参数
	 */
	private CUdeviceptr d_z;
	private CUdeviceptr d_mean;
	private CUdeviceptr d_var;
	private CUdeviceptr d_runingMean;
	private CUdeviceptr d_runingVar;
	
	/**
	 * 反向参数
	 */
	private CUdeviceptr d_dmean;
	private CUdeviceptr d_dvar;


	/**
	 * 前向方法参数
	 */
	private Pointer fastMeanParameters;
	private Pointer fastVarParameters;
	private Pointer normalizeParameters;
	private Pointer normalize_test_Parameters;
	private Pointer mwaParameters;
	
	/**
	 * 反向方法参数
	 */
	private Pointer dgammaParameters;
	private Pointer dbetaParameters;
	private Pointer dbetaFullParameters;
	private Pointer dxhatParameters;
	private Pointer fastDmeanParameters;
	private Pointer fastDvarParameters;
	private Pointer dxParameters;

	public BNKernel3(int C,int H,int W) {
		this.C = C;
		this.H = H;
		this.W = W;
		this.spatial = H * W;
		init();
	}
	
	public void initFunction() {
		
		try {
			
			if(fast_mean_function == null) {
				fast_mean_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "fast_mean_kernel");
			}
			
			if(fast_var_function == null) {
				fast_var_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "fast_variance_kernel");
			}
			
			if(normalize_function == null) {
				normalize_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "normalize_kernel");
			}
			
			if(normalize_test_function == null) {
				normalize_test_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "normalize_test_kernel");
			}
			
			if(mwa_function == null) {
				mwa_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "mwa_kernel");
			}

			if(dgamma_function == null) {
				dgamma_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "backward_scale_kernel");
			}
			
			if(dbeta_function == null) {
				dbeta_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "backward_bias_kernel");
			}
			
			if(dbeta_full_function == null) {
				dbeta_full_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "backward_bias_conn_kernel");
			}
			
			if(dxhat_function == null) {
				dxhat_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "scale_bias_kernel");
			}
			
			if(fast_dmean_function == null) {
				fast_dmean_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "fast_mean_delta_kernel");
			}
			
			if(fast_dvar_function == null) {
				fast_dvar_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "fast_variance_delta_kernel");
			}
			
			if(dx_function == null) {
				dx_function = CUDAModules.getFunctionByModule("H://BNKernel3.cu", "normalize_delta_kernel");
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	private void initKernel() {
//		System.out.println(C);
		/**
		 * 申请向前传播参数显存
		 */
		this.d_mean = CUDAMemoryManager.getDevice(C);
		this.d_var = CUDAMemoryManager.getDevice(C);
		this.d_runingMean = CUDAMemoryManager.getDevice(C);
		this.d_runingVar = CUDAMemoryManager.getDevice(C);
		this.d_dmean = CUDAMemoryManager.getDevice(C);
		this.d_dvar = CUDAMemoryManager.getDevice(C);
		
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();
		
		/**
		 * 申请显存
		 */
		initKernel();

	}
	
	public void initForward(Tensor input,Tensor gama,Tensor beta,Tensor output) {
		
		if(input.number != this.N) {
			
			this.N = input.number;
			
			/**
			 * float *x, int batch, int filters, int spatial, float *mean
			 */
			fastMeanParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {H * W}),
	                Pointer.to(d_mean)
	            );
			
			/**
			 * float *x, float *mean, int batch, int filters, int spatial, float *variance
			 */
			fastVarParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(d_mean),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {H * W}),
	                Pointer.to(d_var)
	            );
			
			/**
			 * float* mean,float* var,float* runingMean,float* runingStd,int n
			 */
			mwaParameters = Pointer.to(
	                Pointer.to(d_mean),
	                Pointer.to(d_var),
	                Pointer.to(d_runingMean),
	                Pointer.to(d_runingVar),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new float[]{momentum})
	            );
			
			d_z = CUDAMemoryManager.getDevice(N * C * H * W);

			/**
			 * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial
			 */
			normalizeParameters = Pointer.to(
					Pointer.to(new int[] {N * C * H * W}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(d_z),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(d_mean),
	                Pointer.to(d_var),
	                Pointer.to(gama.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(new float[] {eta})
	            );
			

			/**
			 * int N, float *x, float *z, float *out, float *mean, float *variance, float *gama, float *beta,int batch, int filters, int spatial
			 */
			normalize_test_Parameters = Pointer.to(
					Pointer.to(new int[] {N * C * H * W}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(output.getGpuData()),
	                Pointer.to(d_runingMean),
	                Pointer.to(d_runingVar),
	                Pointer.to(gama.getGpuData()),
	                Pointer.to(beta.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(new float[] {eta})
	            );

		}
		
	}
	
	public void initBackward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgama,Tensor dbeta) {

		if(dgammaParameters == null) {

//			diff.setGpuData(delta.getGpuData());
			
			/**
			 * float *x_norm, float *delta, int batch, int n, int size, float *scale_updates
			 */
			dgammaParameters = Pointer.to(
	                Pointer.to(d_z),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(dgama.getGpuData())
	            );
			
			/**
			 * float *bias_updates, float *delta, int batch, int n, int size
			 */
			dbetaParameters = Pointer.to(
	                Pointer.to(dbeta.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial})
	            );
			
			/**
			 * float *bias_updates, float *delta, int batch, int n
			 */
			dbetaFullParameters = Pointer.to(
	                Pointer.to(dbeta.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C})
	            );
			
			/**
			 * float *output, float *biases, int n, int size
			 */
			dxhatParameters = Pointer.to(
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(gamma.getGpuData()),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial})
	            );
			
			/**
			 * float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta
			 */
			fastDmeanParameters = Pointer.to(
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(d_var),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(d_dmean)
	            );
			
			/**
			 * float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta
			 */
			fastDvarParameters = Pointer.to(
	                Pointer.to(input.getGpuData()),
	                Pointer.to(diff.getGpuData()),
	                Pointer.to(d_mean),
	                Pointer.to(d_var),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(d_dvar)
	            );
			
			/**
			 * int N, float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta
			 */
			dxParameters = Pointer.to(
					Pointer.to(new int[] {N * C * spatial}),
	                Pointer.to(input.getGpuData()),
	                Pointer.to(d_mean),
	                Pointer.to(d_var),
	                Pointer.to(d_dmean),
	                Pointer.to(d_dvar),
	                Pointer.to(new int[] {N}),
	                Pointer.to(new int[] {C}),
	                Pointer.to(new int[] {spatial}),
	                Pointer.to(diff.getGpuData())
	            );
			
		}
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void forward(RunModel RUN_MODEL, Tensor gama, Tensor beta, Tensor input, Tensor output) {

		initForward(input, gama, beta, output);
		
		if(RUN_MODEL == RunModel.TRAIN) {
			
			/**
			 * 计算标准差
			 * mean = 1/m ∑(x)
			 * var = 1/m ∑(x - mean)^2
			 * std = (var + eta)^1/2
			 */

			fast_mean();

			fast_var();

			/**
			 * 移动加权平均法计算均值与方差
			 */
			mwa();

			normalize_train(input, gama, beta, output);

		}else {

			normalize_test(input, gama, beta, output);
			
		}

	}
	
	public void fast_mean() {
		
		try {

	        cuLaunchKernel(fast_mean_function,
	        		C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            fastMeanParameters, null // Kernel- and extra parameters
		        );
	        
//	        JCudaDriver.cuCtxSynchronize();
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void fast_var() {
		
		try {

	        cuLaunchKernel(fast_var_function,
	        		C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            fastVarParameters, null // Kernel- and extra parameters
		        );
	        
//	        JCudaDriver.cuCtxSynchronize();
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void normalize_test(Tensor input,Tensor gama, Tensor beta, Tensor output) {
		
		try {
			
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

	public void normalize_train(Tensor input,Tensor gama, Tensor beta, Tensor output) {
		
		try {

	        cuLaunchKernel(normalize_function,
		            this.CAFFE_GET_BLOCKS(N * C * H * W),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            normalizeParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mwa() {
		
		try {

	        cuLaunchKernel(mwa_function,
		            this.CAFFE_GET_BLOCKS(C),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            mwaParameters, null // Kernel- and extra parameters
		        );
	        
//	        JCudaDriver.cuCtxSynchronize();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor input,Tensor delta,Tensor diff,Tensor gamma,Tensor dgamma,Tensor dbeta) {

		initBackward(input, delta, diff, gamma, dgamma, dbeta);

		if(spatial == 1) {
			dbetaFull();
		}else {
			dbeta();
//			showDM(dbeta.getGpuData(), C);
		}
		
		dgamma();
		
		dxhat();
		
		fastDmean();
		
		fastDvar();
//
//		showDM(d_dvar, C);
//		showDM(d_dmean, C);
//		
		dx();

//		showDM(diff.getGpuData(), diff.dataLength);
		
	}
	
	public void dgamma() {

		try {

	        cuLaunchKernel(dgamma_function,
		            C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            dgammaParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void dbeta() {

		try {

	        cuLaunchKernel(dbeta_function,
		            C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            dbetaParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void dbetaFull() {

		try {

	        cuLaunchKernel(dbeta_full_function,
		            this.CAFFE_GET_BLOCKS(C),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            dbetaFullParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void dxhat() {

		try {

	        cuLaunchKernel(dxhat_function,
		            (spatial - 1) / CAFFE_CUDA_NUM_THREADS + 1,  C, N,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            dxhatParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void fastDmean() {

		try {

	        cuLaunchKernel(fast_dmean_function,
		            C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            fastDmeanParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void fastDvar() {

		try {

	        cuLaunchKernel(fast_dvar_function,
		            C,  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            fastDvarParameters, null // Kernel- and extra parameters
		        );
	        
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void dx() {

		try {
			
	        cuLaunchKernel(dx_function,
	        		this.CAFFE_GET_BLOCKS(N * C * spatial),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            dxParameters, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
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
    
    public static void main(String args[]) {
    	
    	CUDAModules.initContext();
    	
    	test2d();
    	
    	test2d_cpu();
    	
    }
    
    public static void test2d() {
    	int N = 2;
    	int C = 3;
    	int H = 5;
    	int W = 5;
    	
//    	float[] x = RandomUtils.order(N * C * H * W, 1.0f, 1f);
    	
    	float[] x = new float[] {0.9827f, 0.5268f, 0.4057f, 0.2853f, 0.1708f,
        0.4791f, 0.5626f, 0.1290f, 0.9540f, 0.7471f,
        0.5806f, 0.8789f, 0.9766f, 0.8142f, 0.9557f,
        0.2814f, 0.7667f, 0.5963f, 0.0016f, 0.5944f,
        0.4617f, 0.0975f, 0.3558f, 0.3318f, 0.5196f,
       	0.7558f, 0.7438f, 0.4061f, 0.2737f, 0.1826f,
        0.7600f, 0.3608f, 0.3924f, 0.2537f, 0.7536f,
        0.7980f, 0.5246f, 0.6428f, 0.0571f, 0.9973f,
        0.7106f, 0.5854f, 0.3122f, 0.2741f, 0.2868f,
        0.4628f, 0.2696f, 0.0436f, 0.1222f, 0.4933f,
       	0.5372f, 0.4992f, 0.2837f, 0.8462f, 0.2095f,
        0.1916f, 0.1830f, 0.1934f, 0.8305f, 0.0776f,
        0.9014f, 0.1835f, 0.7673f, 0.0999f, 0.5783f,
        0.7816f, 0.2961f, 0.9230f, 0.3454f, 0.6030f,
        0.4821f, 0.0113f, 0.9629f, 0.8698f, 0.8440f,
		0.9763f, 0.7661f, 0.2085f, 0.4248f, 0.7407f,
        0.5092f, 0.5272f, 0.8521f, 0.1649f, 0.9759f,
        0.9084f, 0.3206f, 0.3061f, 0.9648f, 0.3377f,
        0.6753f, 0.6662f, 0.4570f, 0.9556f, 0.0918f,
        0.8788f, 0.6432f, 0.4928f, 0.8778f, 0.5665f,
       	0.7979f, 0.5639f, 0.5970f, 0.4987f, 0.1227f,
        0.4963f, 0.6865f, 0.5728f, 0.1927f, 0.1199f,
        0.5015f, 0.0221f, 0.0826f, 0.0077f, 0.0568f,
        0.7569f, 0.7684f, 0.1536f, 0.4406f, 0.2919f,
        0.3006f, 0.9501f, 0.1994f, 0.3314f, 0.5612f,
       	0.3303f, 0.8773f, 0.3262f, 0.1926f, 0.8667f,
        0.3360f, 0.5357f, 0.3332f, 0.2044f, 0.5538f,
        0.0607f, 0.2203f, 0.7994f, 0.6357f, 0.6469f,
        0.8163f, 0.7764f, 0.6821f, 0.6798f, 0.0553f,
        0.0609f, 0.2305f, 0.7183f, 0.8135f, 0.7688f};
    	
    	float[] d = RandomUtils.val(N * C * H * W, 1.0f);
    	
    	float[] g = MatrixUtils.one(C);
    	
    	Tensor input = new Tensor(N, C, H, W, x, true);
    	Tensor gamma = new Tensor(1, 1, 1, C, g, true);
    	Tensor beta = new Tensor(1, 1, 1, C, true);
    	
    	Tensor delta = new Tensor(N, C, H, W, d, true);
    	
    	Tensor output = new Tensor(N, C, H, W, true);
    	Tensor diff = new Tensor(N, C, H, W, true);
    	
    	Tensor dgamma = new Tensor(1, 1, 1, C, true);
    	Tensor dbeta = new Tensor(1, 1, 1, C, true);
    	
    	BNKernel3 kernel = new BNKernel3(C, H, W);
    	
    	for(int i = 0;i<1;i++) {

        	kernel.forward(RunModel.TRAIN, gamma, beta, input, output);
        	
        	kernel.backward(input, delta, diff, gamma, dgamma, dbeta);
        	
    	}
    	
    	output.syncHost();
    	diff.syncHost();
    	dgamma.syncHost();
    	dbeta.syncHost();
    	
    	System.out.println("=======input==============");
    	
    	PrintUtils.printImage(input.data);
    	
    	System.out.println("");
    	
    	System.out.println("=======output==============");
    	
    	PrintUtils.printImage(output.data);
    	
    	System.out.println("");
    	
    	System.out.println("=======diff==============");
    	
    	PrintUtils.printImage(diff.data);
    	
    	System.out.println("");
    	
    	System.out.println("=======dgamma==============");
    	
    	PrintUtils.printImage(dgamma.data);
    	
    	System.out.println("");
    	
    	System.out.println("=======dbeta==============");
    	
    	PrintUtils.printImage(dbeta.data);
    	
    }
    
    public static void test1d() {
    	int N = 2;
    	int C = 1;
    	int H = 1;
    	int W = 10;
    	
//    	float[] x = RandomUtils.order(N * C * H * W, 0.1f, 0.1f);
    	
    	float[] x = new float[] {56.773f,-7.231f,39.634f,24.728f,-17.959f,55.251f,-52.316f,-36.322f,-29.619f,
    			55.24f,26.773f,-1.231f,19.634f,4.728f,7.958f,-65.251f,52.316f,-36.322f,-23.619f,-5.247f};
    	
    	float[] d = RandomUtils.val(N * C * H * W, 1.0f);
    	
    	float[] g = MatrixUtils.one(W);
    	
    	Tensor input = new Tensor(N, C, H, W, x, true);
    	Tensor gamma = new Tensor(1, 1, 1, W, g, true);
    	Tensor beta = new Tensor(1, 1, 1, W, true);
    	
    	Tensor delta = new Tensor(N, C, H, W, d, true);
    	
    	Tensor output = new Tensor(N, C, H, W, true);
    	Tensor diff = new Tensor(N, C, H, W, true);
    	
    	Tensor dgamma = new Tensor(1, 1, 1, W, true);
    	Tensor dbeta = new Tensor(1, 1, 1, W, true);
    	
    	BNKernel3 kernel = new BNKernel3(W, 1, 1);
    	
    	for(int i = 1;i<2;i++) {

        	kernel.forward(RunModel.TRAIN, gamma, beta, input, output);

//        	float[] d2 = RandomUtils.val(N * C * H * W, i * 1f);
//        	delta.data = d2;
//        	delta.hostToDevice();
        	
        	kernel.backward(input, delta, diff, gamma, dgamma, dbeta);
        	
        	output.syncHost();
        	diff.syncHost();
        	dgamma.syncHost();
        	dbeta.syncHost();
        	
        	System.out.println("");
        	
        	System.out.println("=======input==============");
        	
        	PrintUtils.printImage(input.data);
        	
        	System.out.println("");
        	
        	System.out.println("=======output==============");
        	
        	PrintUtils.printImage(output.data);
        	
        	System.out.println("");
        	
        	System.out.println("=======diff==============");
        	
        	PrintUtils.printImage(diff.data);
        	
        	System.out.println("");
        	
        	System.out.println("=======dgamma==============");
        	
        	PrintUtils.printImage(dgamma.data);
        	
        	System.out.println("");
        	
        	System.out.println("=======dbeta==============");
        	
        	PrintUtils.printImage(dbeta.data);
        	
    	}
    	
    }
    
    public static void test1d_cpu() {
    	
    	int N = 2;
    	int C = 1;
    	int H = 1;
    	int W = 10;
    	
    	int size = N * C * H * W;
    	
    	float[] x = new float[] {56.773f,-7.231f,39.634f,24.728f,-17.959f,55.251f,-52.316f,-36.322f,-29.619f,
    			55.24f,26.773f,-1.231f,19.634f,4.728f,7.958f,-65.251f,52.316f,-36.322f,-23.619f,-5.247f};
    	
    	float[] d = RandomUtils.val(N * C * H * W, 1.0f);
    	
    	float[] g = MatrixUtils.one(W);
    	
    	Tensor input = new Tensor(N, C, H, W, x);
    	Tensor gamma = new Tensor(1, 1, 1, W, g);
    	Tensor beta = new Tensor(1, 1, 1, W);
    	
    	Tensor output = new Tensor(N, C, H, W);
    	
    	BNKernel3 kernel = new BNKernel3(W, 1, 1);
    	
    	float[] mean = new float[W];
    	float[] var = new float[W];
    	float[] runingMean = new float[W];
    	float[] runingVar = new float[W];
    	
    	float[] z = new float[size];
    	
    	kernel.N = N;
    	
    	kernel.mean_cpu(x, mean);
    	kernel.variance_cpu(x, mean, var);
    	
    	kernel.mwa_cpu(mean, var, runingMean, runingVar);
    	
    	kernel.normalize_cpu(x, z, output.data, mean, var, gamma.data, beta.data);
    	
    	System.out.println("");
    	
    	System.out.println(JsonUtils.toJson(output.data));
    	
    	float[] dgamma = new float[kernel.C];
    	float[] dbeta = new float[kernel.C];
    	
    	kernel.backward_cpu(d, dgamma, dbeta, x, gamma.data, z, mean, var);
    	
    	System.out.println(JsonUtils.toJson(dgamma));
    	
    	System.out.println(JsonUtils.toJson(dbeta));
    	
    	System.out.println(JsonUtils.toJson(d));
    	
    }
    
   public static void test2d_cpu() {
    	
	   	int N = 2;
		int C = 3;
		int H = 5;
		int W = 5;
		
		int size = N * C * H * W;
		
		//   	float[] x = RandomUtils.order(N * C * H * W, 1.0f, 1f);
		
		float[] x = new float[] {0.9827f, 0.5268f, 0.4057f, 0.2853f, 0.1708f,
		   0.4791f, 0.5626f, 0.1290f, 0.9540f, 0.7471f,
		   0.5806f, 0.8789f, 0.9766f, 0.8142f, 0.9557f,
		   0.2814f, 0.7667f, 0.5963f, 0.0016f, 0.5944f,
		   0.4617f, 0.0975f, 0.3558f, 0.3318f, 0.5196f,
		  	0.7558f, 0.7438f, 0.4061f, 0.2737f, 0.1826f,
		   0.7600f, 0.3608f, 0.3924f, 0.2537f, 0.7536f,
		   0.7980f, 0.5246f, 0.6428f, 0.0571f, 0.9973f,
		   0.7106f, 0.5854f, 0.3122f, 0.2741f, 0.2868f,
		   0.4628f, 0.2696f, 0.0436f, 0.1222f, 0.4933f,
		  	0.5372f, 0.4992f, 0.2837f, 0.8462f, 0.2095f,
		   0.1916f, 0.1830f, 0.1934f, 0.8305f, 0.0776f,
		   0.9014f, 0.1835f, 0.7673f, 0.0999f, 0.5783f,
		   0.7816f, 0.2961f, 0.9230f, 0.3454f, 0.6030f,
		   0.4821f, 0.0113f, 0.9629f, 0.8698f, 0.8440f,
			0.9763f, 0.7661f, 0.2085f, 0.4248f, 0.7407f,
		   0.5092f, 0.5272f, 0.8521f, 0.1649f, 0.9759f,
		   0.9084f, 0.3206f, 0.3061f, 0.9648f, 0.3377f,
		   0.6753f, 0.6662f, 0.4570f, 0.9556f, 0.0918f,
		   0.8788f, 0.6432f, 0.4928f, 0.8778f, 0.5665f,
		  	0.7979f, 0.5639f, 0.5970f, 0.4987f, 0.1227f,
		   0.4963f, 0.6865f, 0.5728f, 0.1927f, 0.1199f,
		   0.5015f, 0.0221f, 0.0826f, 0.0077f, 0.0568f,
		   0.7569f, 0.7684f, 0.1536f, 0.4406f, 0.2919f,
		   0.3006f, 0.9501f, 0.1994f, 0.3314f, 0.5612f,
		  	0.3303f, 0.8773f, 0.3262f, 0.1926f, 0.8667f,
		   0.3360f, 0.5357f, 0.3332f, 0.2044f, 0.5538f,
		   0.0607f, 0.2203f, 0.7994f, 0.6357f, 0.6469f,
		   0.8163f, 0.7764f, 0.6821f, 0.6798f, 0.0553f,
		   0.0609f, 0.2305f, 0.7183f, 0.8135f, 0.7688f};
    	
    	float[] d = RandomUtils.val(N * C * H * W, 1.0f);
    	
    	float[] g = MatrixUtils.one(C);
    	
    	Tensor input = new Tensor(N, C, H, W, x);
    	Tensor gamma = new Tensor(1, 1, 1, C, g);
    	Tensor beta = new Tensor(1, 1, 1, C);
    	
    	Tensor output = new Tensor(N, C, H, W);
    	
    	BNKernel3 kernel = new BNKernel3(C, H, W);
    	
    	float[] mean = new float[C];
    	float[] var = new float[C];
    	float[] runingMean = new float[C];
    	float[] runingVar = new float[C];
    	
    	float[] z = new float[size];
    	
    	kernel.N = N;
    	
    	kernel.mean_cpu(x, mean);
    	kernel.variance_cpu(x, mean, var);
    	
    	kernel.mwa_cpu(mean, var, runingMean, runingVar);
    	
    	kernel.normalize_cpu(x, z, output.data, mean, var, gamma.data, beta.data);
    	
    	System.out.println("");
    	
    	System.out.println(JsonUtils.toJson(output.data));
    	
    	float[] dgamma = new float[kernel.C];
    	float[] dbeta = new float[kernel.C];
    	
    	kernel.backward_cpu(d, dgamma, dbeta, x, gamma.data, z, mean, var);
    	
    	System.out.println(JsonUtils.toJson(dgamma));
    	
    	System.out.println(JsonUtils.toJson(dbeta));
    	
    	System.out.println(JsonUtils.toJson(d));
    	
    }
    
    public void backward_cpu(float[] delta,float[] dgamma,float[] dbeta,float[] x,float[] gamma,float[] z,float[] mean,float[] var) {
    	
    	float[] dmean = new float[C];
    	float[] dvar = new float[C];
    	
    	dbeta_cpu(delta, dbeta);
    	
    	dgamma_cpu(dgamma, delta, z);
    	
    	dz_cpu(delta, gamma);
    	
    	dmean_cpu(delta, dmean, var);

    	dvar_cpu(dvar, delta, mean, var, x);
    	
    	System.out.println("dvar:"+JsonUtils.toJson(dvar));
    	System.out.println("dmean:"+JsonUtils.toJson(dmean));

    	dx_cpu(delta, var, dvar, mean, dmean, x);
    	
    }
    
    public void mean_cpu(float[] x,float[] mean) {
    	
    	float scale = 1.0f/(N * spatial);

        int i,j,k;
        for(i = 0; i < C; ++i){
            mean[i] = 0;
            for(j = 0; j < N; ++j){
                for(k = 0; k < spatial; ++k){
                    int index = j*C*spatial + i*spatial + k;
                    mean[i] += x[index];
                }
            }
            mean[i] *= scale;
        }
    	
    }
    
    public void variance_cpu(float[] x, float[] mean, float[] variance){
        float scale = 1.0f/(N * spatial - 1);
        int i,j,k;
        for(i = 0; i < C; ++i){
            variance[i] = 0;
            for(j = 0; j < N; ++j){
                for(k = 0; k < spatial; ++k){
                    int index = j*C*spatial + i*spatial + k;
                    variance[i] += Math.pow((x[index] - mean[i]), 2);
                }
            }
            variance[i] *= scale;
        }
    }
    
    public void mwa_cpu(float[] mean,float[] var,float[] runingMean,float[] runingVar) {
    	for(int index = 0;index<C;index++){
    		runingMean[index] = (1.0f - momentum) * runingMean[index] + momentum * mean[index];
    		runingVar[index] = (1.0f - momentum) * runingVar[index] + momentum * var[index];
    	}
    }
    
    public void normalize_cpu(float[] x,float[] z,float[] output,float[] mean, float[] variance,float[] gamma,float[] beta) {
    	int b, f, i;
        for(b = 0; b < N; ++b){
            for(f = 0; f < C; ++f){
                for(i = 0; i < spatial; ++i){
                    int index = b*C*spatial + f*spatial + i;
                    z[index] = (x[index] - mean[f])/((float)Math.sqrt(variance[f]) + .000001f);
                    output[index] = gamma[f] * z[index] + beta[f];
                }
            }
        }
    }
    
    public void dbeta_cpu(float[] delta,float[] dbeta) {
    	for(int index = 0;index<C;index++) {
    		int b;
            float sum = 0;
            for(b = 0; b < N; ++b){
                for(int j = 0;j<spatial;j++) {
                	sum += delta[b * C * spatial + index * spatial + j];
                }
            }
            dbeta[index] += sum;
    	}
    }
    
    public void dgamma_cpu(float[] dgamma,float[] delta,float[] z) {
    	int i,b,f;
        for(f = 0; f < C; ++f){
            float sum = 0;
            for(b = 0; b < N; ++b){
                for(i = 0; i < spatial; ++i){
                    int index = i + spatial*(f + C*b);
                    sum += delta[index] * z[index];
                }
            }
            dgamma[f] += sum;
        }
    }
    
    public void dz_cpu(float[] delta,float[] gamma) {
    	int i,j,b;
        for(b = 0; b < N; ++b){
            for(i = 0; i < C; ++i){
                for(j = 0; j < spatial; ++j){
                	delta[(b*C + i)*spatial + j] *= gamma[i];
                }
            }
        }
    }
    
    public void dmean_cpu(float[] delta,float[] dmean,float[] var) {
        int i,j,k;
        for(i = 0; i < C; ++i){
            dmean[i] = 0;
            for (j = 0; j < N; ++j) {
                for (k = 0; k < spatial; ++k) {
                    int index = j*C*spatial + i*spatial + k;
                    dmean[i] += delta[index];
                }
            }
            dmean[i] *= (-1./Math.sqrt(var[i] + .00001f));
        }
    }
    
    public void dvar_cpu(float[] dvar,float[] delta,float[] mean,float[] var,float[] x) {
    	int i,j,k;
	    for(i = 0; i < C; ++i){
	        dvar[i] = 0;
	        for(j = 0; j < N; ++j){
	            for(k = 0; k < spatial; ++k){
	                int index = j*C*spatial + i*spatial + k;
	                dvar[i] += delta[index]*(x[index] - mean[i]);
	            }
	        }
	        dvar[i] *= -.5f * Math.pow(var[i] + .00001f, -1.5);
	    }
    }
    
    public void dx_cpu(float[] delta,float[] var,float[] dvar,float[] mean,float[] dmean,float[] x) {
    	int f, j, k;
        for(j = 0; j < N; ++j){
            for(f = 0; f < C; ++f){
                for(k = 0; k < spatial; ++k){
                    int index = j*C*spatial + f*spatial + k;
//                    System.out.print(delta[index] * 1.0f/(Math.sqrt(var[f] + .00001f)));
//                    System.out.print("-");
//                    System.out.println(dvar[f] * 2.0f * (x[index] - mean[f]) / (spatial * N) + dmean[f]/(spatial * N));
                    delta[index] = (float) (delta[index] * 1.0f/(Math.sqrt(var[f] + .00001f)) + dvar[f] * 2.0f * (x[index] - mean[f]) / (spatial * N) + dmean[f]/(spatial * N));
                }
            }
        }
    }
    
}
