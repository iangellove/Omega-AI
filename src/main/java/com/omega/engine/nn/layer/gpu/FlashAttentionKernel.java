package com.omega.engine.nn.layer.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class FlashAttentionKernel extends BaseKernel{

	private CUfunction forward_function;
	
	private CUfunction forward2_function;
	
	private Pointer kernelParameters;

	private Tensor d_m;
	private Tensor d_l;
	
	private int headNum;
	
	private int time;
	
	private int headDim;
	
	public FlashAttentionKernel(int headNum,int time,int headDim) {
		this.headNum = headNum;
		this.time = time;
		this.headDim = headDim;
		init();
	}
	
	public void init() {
		/**
		 * 初始化cuda函数
		 */
		initFunction();

	}
	
	private void initKernel(Tensor input,int headNum,int time) {
		
		if(input.number != N) {
			N = input.number;
			/**
			 * 申请向前传播参数显存
			 */
			this.d_m = new Tensor(input.number, headNum, time, 1, MatrixUtils.val(input.number * headNum * time, Float.NEGATIVE_INFINITY), true);
			
			this.d_l = new Tensor(input.number, headNum, time, 1, true);

		}
		
	}
	
	public void initFunction() {
		
		try {

			if(forward_function == null) {

				forward_function = CUDAModules.getLocalFunctionByModule("FlashAttentionKernel.cu", "forward_kernel");
				
			}
			
			if(forward2_function == null) {

//				forward2_function = CUDAModules.getLocalFunctionByModule("FlashAttentionKernel.cu", "flash_attention_v1_kernel");
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void forward(Tensor Q,Tensor K,Tensor V,Tensor output) {
		
		try {
			
			initKernel(Q, headNum, time);
			
			int B = Q.number;
			int nh = headNum;
			int N = time;
			int d = headDim;

//			int Bc = (int) Math.ceil(CUDAModules.props.sharedMemPerBlock / 4 / d);
//			int Br = (int) Math.ceil(Math.min(CUDAModules.props.sharedMemPerBlock / 4 / d, d));
			
			int Bc = 32;
			int Br = 32;
			
			float scale = (float) (1.0f / Math.sqrt(d));
			int Tc = (int) Math.ceil((float) N / Bc);
			int Tr = (int) Math.ceil((float) N / Br);
			System.out.println(Bc);
			System.out.println(Br);
			System.out.println(Tc);
			System.out.println(Tr);
	        /**
	         * 设置入参
	         * const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O
	         */ 
			kernelParameters = Pointer.to(
	        		Pointer.to(Q.getGpuData()),
	        		Pointer.to(K.getGpuData()),
	        		Pointer.to(V.getGpuData()),
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(new int[]{d}),
	        		Pointer.to(new int[]{Tc}),
	        		Pointer.to(new int[]{Tr}),
	        		Pointer.to(new int[]{Bc}),
	        		Pointer.to(new int[]{Br}),
	        		Pointer.to(new float[]{scale}),
	                Pointer.to(d_l.getGpuData()),
	                Pointer.to(d_m.getGpuData()),
	                Pointer.to(output.getGpuData())
	            );
	        
			int sram_size = (3 * Bc * d * Sizeof.FLOAT) + (Bc * Br * Sizeof.FLOAT);
			
			int[] grid_dim = new int[] {B, nh, 1};
			int[] block_dim = new int[] {Bc, 1, 1};
			
			System.out.println("max share memory size:"+CUDAModules.props.sharedMemPerBlock + ".current size:"+sram_size);
			
		    checkCUDA(cuLaunchKernel(forward_function,
		    		grid_dim[0],  grid_dim[1], grid_dim[2],      // Grid dimension
		    		block_dim[0], block_dim[1], block_dim[2],      // Block dimension
		            sram_size, null,               // Shared memory size and stream
		            kernelParameters, null // Kernel- and extra parameters
		        ));

//		    d_m.showDM();
//		    d_l.showDM();
		    
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void backward(Tensor Q,Tensor K,Tensor V,Tensor diff,Tensor dQ,Tensor dK,Tensor dV) {
		
		try {
			
//			initKernel(Q, headNum, time);
			
			int B = Q.number;
			int nh = headNum;
			int N = time;
			int d = headDim;

//			int Bc = (int) Math.ceil(CUDAModules.props.sharedMemPerBlock / 4 / d);
//			int Br = (int) Math.ceil(Math.min(CUDAModules.props.sharedMemPerBlock / 4 / d, d));
			
			int Bc = 16;
			int Br = 16;
			
			float scale = (float) (1.0f / Math.sqrt(d));
			int Tc = (int) Math.ceil((float) N / Bc);
			int Tr = (int) Math.ceil((float) N / Br);
			System.out.println(Bc);
			System.out.println(Br);
			System.out.println(Tc);
			System.out.println(Tr);
	        /**
	         * 设置入参
	         * const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O
	         */ 
//			kernelParameters = Pointer.to(
//	        		Pointer.to(Q.getGpuData()),
//	        		Pointer.to(K.getGpuData()),
//	        		Pointer.to(V.getGpuData()),
//	        		Pointer.to(new int[]{N}),
//	        		Pointer.to(new int[]{d}),
//	        		Pointer.to(new int[]{Tc}),
//	        		Pointer.to(new int[]{Tr}),
//	        		Pointer.to(new int[]{Bc}),
//	        		Pointer.to(new int[]{Br}),
//	        		Pointer.to(new float[]{scale}),
//	                Pointer.to(d_l.getGpuData()),
//	                Pointer.to(d_m.getGpuData()),
//	                Pointer.to(output.getGpuData())
//	            );
//	        
			int sram_size = ((3 * Bc * d * Sizeof.FLOAT) + (Bc * Br * Sizeof.FLOAT)) * 2;
			
			int[] grid_dim = new int[] {B, nh, 1};
			int[] block_dim = new int[] {Bc, 1, 1};
			
			System.out.println("max share memory size:"+CUDAModules.props.sharedMemPerBlock + ".current size:"+sram_size);
			
//		    checkCUDA(cuLaunchKernel(forward_function,
//		    		grid_dim[0],  grid_dim[1], grid_dim[2],      // Grid dimension
//		    		block_dim[0], block_dim[1], block_dim[2],      // Block dimension
//		            sram_size, null,               // Shared memory size and stream
//		            kernelParameters, null // Kernel- and extra parameters
//		        ));

		    d_m.showDM();
		    d_l.showDM();
		    
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
//	public void forward2(Tensor Q,Tensor K,Tensor V,Tensor output) {
//		
//		try {
//			
//			initKernel(Q, headNum, time);
//			
//			int B = Q.number;
//			int nh = headNum;
//			int N = time;
//			int d = headDim;
//
//			int Bc = 2;
//			int Br = 2;
//			
//			float scale = (float) (1.0f / Math.sqrt(d));
//			int Tc = (int) Math.ceil((float) N / Bc);
//			int Tr = (int) Math.ceil((float) N / Br);
//			System.out.println(Bc);
//			System.out.println(Br);
//			System.out.println(Tc);
//			System.out.println(Tr);
//	        /**
//	         * 设置入参
//	         * float *Q, float *K, float *V, float *O, float *gMax,float *gDenom, int seqlen, int stride_head, float smScale,int kBc,int kBr,int Kdim
//	         */ 
//			kernelParameters = Pointer.to(
//	        		Pointer.to(Q.getGpuData()),
//	        		Pointer.to(K.getGpuData()),
//	        		Pointer.to(V.getGpuData()),
//	        		Pointer.to(output.getGpuData()),
//	        		Pointer.to(d_l.getGpuData()),
//	                Pointer.to(d_m.getGpuData()),
//	        		Pointer.to(new int[]{N}),
//	        		Pointer.to(new int[]{N * d}),
//	        		Pointer.to(new float[]{scale}),
//	        		Pointer.to(new int[]{Bc}),
//	        		Pointer.to(new int[]{Br}),
//	        		Pointer.to(new int[]{d})
//	            );
//	        
//			int Gc = B * nh;
//			int Gr = (N + Br - 1) / Br;
//			
//			int[] grid_dim = new int[] {Gc, Gr, 1};
//			int[] block_dim = new int[] {Bc, Br, 1};
//			
//			/**
//			 * int kBc, int kBr, int kDim
//			 */
//			Pointer extra = Pointer.to(
//						Pointer.to(new int[]{Bc}),
//		        		Pointer.to(new int[]{Br}),
//		        		Pointer.to(new int[]{d})
//					);
//			
//		    checkCUDA(cuLaunchKernel(forward2_function,
//		    		grid_dim[0],  grid_dim[1], grid_dim[2],      // Grid dimension
//		    		block_dim[0], block_dim[1], block_dim[2],      // Block dimension
//		            0, null,               // Shared memory size and stream
//		            kernelParameters, extra // Kernel- and extra parameters
//		        ));
//
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//		}
//		
//	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
			throw new RuntimeException("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public static void main(String[] args) {
		
		CUDAModules.initContext();
		
		int batchSize = 16;
		int headNum = 12;
		int time = 64;
		int headDim = 64; //headDim
		int len = batchSize * headNum * time * headDim;
		
		Tensor Q = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
		Tensor K = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
		Tensor V = new Tensor(batchSize, headNum, time, headDim, MatrixUtils.order(len, 0.01f, 0.1f), true);
		
		Tensor output = new Tensor(batchSize, headNum, time, headDim, true);
		
		FlashAttentionKernel kernel = new FlashAttentionKernel(headNum, time, headDim);
		
		for(int i = 0;i<10;i++) {
			long startTime = System.nanoTime();
		    
			kernel.forward(Q, K, V, output);

			output.showDM(0);
			
			System.out.println((System.nanoTime() - startTime)/1e6+"ms.");
		}
//		
//		kernel.backward(Q, K, V, null, null, null, null);

//		test();
		
//		kernel.forward2(Q, K, V, output);
		
		
		
	}
	
	public static void test() {
		
		int Tr = 2;
		int Tc = 2;
		int Br = 2;
		int Bc = 2;
		int d = 3;
		int N = 4;
		int tile_size = Bc * d;
		
		int len = 1 * 1 * N * d;
		
		float[] Q = MatrixUtils.order(len, 1f, 1f);
		float[] K = MatrixUtils.order(len, 1f, 1f);
		
		float[] Qi = new float[Br * d];
		float[] Kj = new float[Bc * d];
		float[] S = new float[Bc * d];
		

		for (int j = 0; j < Tc; j++) {
			
			for(int tx = 0;tx<Bc;tx++) {
		        // Load Kj, Vj to SRAM
		        for (int x = 0; x < d; x++) {
		        	if(j * Bc + tx < N){
		        		Kj[(tx * d) + x] = K[(tile_size * j) + (tx * d) + x];
		        	}else{
		        		Kj[(tx * d) + x] = 0;
		        	}
		        }
			}
			
	        System.out.println("j:["+j+"]Kj:"+JsonUtils.toJson(Kj));
	        
	        for(int tx = 0;tx<Bc;tx++) {
	        	
		        for (int i = 0; i < Tr; i++)  {
	
		            if((Br * i) + tx >= N){
		            	break;
		            }
		            
		            // Load Qi to SRAM, l and m to registers
		            for (int x = 0; x < d; x++) {
		            	if(i * Br + tx < N){
		                	Qi[(tx * d) + x] = Q[(tile_size * i) + (tx * d) + x];
		                }else{
		                	Qi[(tx * d) + x] = 0.0f;
		                }
		            }
		            
		            System.out.println("i:["+i+"]Qi:"+JsonUtils.toJson(Qi));
		            
		            // S = QK^T, row_m = rowmax(S)
		            for (int y = 0; y < Bc; y++) {
		                float sum = 0;
		                for (int x = 0; x < d; x++) {
	//	                	System.out.println(((tx * d) + x) + "*" + ((y * d) + x));
	//	                	System.out.println(Qi[(tx * d) + x] + "*" + Kj[(y * d) + x]);
		                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
		                }
		                S[(Bc * tx) + y] = sum;
		                System.out.println("sum:"+sum);
		            }
		            
		            System.out.println("S:"+JsonUtils.toJson(S));
		        }
	        }
		}
		
		
	}
	
}
