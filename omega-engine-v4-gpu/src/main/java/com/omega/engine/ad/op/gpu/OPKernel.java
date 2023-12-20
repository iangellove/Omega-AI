package com.omega.engine.ad.op.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import java.io.Serializable;

import com.omega.common.data.Tensor;
import com.omega.common.lib.LibPaths;
import com.omega.engine.gpu.CUDAModules;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.cudaError;

public class OPKernel implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3345793649705471080L;

	public int N = 0;
	
	private static OPKernel kernel = null;

	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUfunction fill_gpu_function;
	
	private CUfunction axpy_gpu_function;
	
	private CUfunction copy_gpu_function;
	
	private CUfunction copy_number_gpu_function;
	
	private CUfunction copy_channel_gpu_function;
	
	private CUfunction add_gpu_function;
	
	private CUfunction add_scalar_gpu_function;
	
	private CUfunction add_number_gpu_function;
	
	private CUfunction add_channel_gpu_function;
	
	private CUfunction sub_gpu_function;
	
	private CUfunction sub_scalar_gpu_function;
	
	private CUfunction scalar_sub_gpu_function;
	
	private CUfunction mul_gpu_function;
	
	private CUfunction mul_scalar_gpu_function;
	
	private CUfunction mul_plus_gpu_function;
	
	private CUfunction mul_plus_scalar_gpu_function;
	
	private CUfunction div_gpu_function;
	
	private CUfunction div_scalar_gpu_function;
	
	private CUfunction scalar_div_gpu_function;
	
	private CUfunction div_plus_gpu_function;
	
	private CUfunction div_plus_scalar_gpu_function;
	
	private CUfunction div_bGrad_gpu_function;
	
	private CUfunction div_scalar_bGrad_gpu_function;
	
	private CUfunction scalar_plus_div_gpu_function;
	
	private CUfunction pow_gpu_function;
	
	private CUfunction log_gpu_function;
	
	private CUfunction exp_gpu_function;
	
	private CUfunction sin_gpu_function;
	
	private CUfunction cos_gpu_function;
	
	private CUfunction tan_gpu_function;
	
	private CUfunction tan_back_gpu_function;
	
	private CUfunction atan_gpu_function;
	
	private CUfunction atan_back_gpu_function;
	
	private CUfunction sum_gpu_function;
	
	private CUfunction sum_channel_gpu_function;
	
	private CUfunction broadcast_gpu_function;

	private CUfunction broadcast_channel_gpu_function;
	
	private CUfunction clamp_gpu_function;
	
	private CUfunction clamp_back_gpu_function;
	
	private CUfunction maximum_gpu_function;
	
	private CUfunction minimum_gpu_function;
	
	private CUfunction maximum_back_gpu_function;
	
	private CUfunction minimum_back_gpu_function;
	
	
	
	public OPKernel() {
		
		fill_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "fill_kernel");
		
		axpy_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "axpy_kernel"); 
		
		copy_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "copy_kernel");
		
		copy_number_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "copy_number_kernel");
		
		copy_channel_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "copy_channel_kernel");
		
		add_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "add_kernel");
		
		add_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "add_scalar_kernel");
		
		add_number_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "add_number_kernel");
		
		add_channel_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "add_channel_kernel");

		sub_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "sub_kernel");
		
		sub_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "sub_scalar_kernel");
		
		scalar_sub_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "scalar_sub_kernel"); 
		
		mul_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "mul_kernel");
		
		mul_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "mul_scalar_kernel");
		
		mul_plus_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "mul_plus_kernel");
		
		mul_plus_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "mul_plus_scalar_kernel");
		
		div_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_kernel");
		
		div_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_scalar_kernel");
		
		scalar_div_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "scalar_div_kernel");
		
		div_plus_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_plus_kernel");
		
		div_plus_scalar_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_plus_scalar_kernel");
		
		scalar_plus_div_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "scalar_plus_div_kernel");
		
		div_bGrad_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_bGrad_kernel");
		
		div_scalar_bGrad_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "div_scalar_bGrad_kernel");
		
		pow_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "pow_kernel");
		
		log_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "log_kernel");
		
		exp_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "exp_kernel");
		
		sin_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "sin_kernel");
		
		cos_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "cos_kernel");
		
		tan_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "tan_kernel");
		
		atan_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "atan_kernel");
		
		tan_back_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "tan_back_kernel");
		
		atan_back_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "atan_back_kernel");
		
		sum_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "sum_kernel");
		
		sum_channel_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "sum_channel_kernel");
		
		broadcast_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "broadcast_kernel");
		
		broadcast_channel_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "broadcast_number_kernel");
		
		clamp_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "clamp_kernel");
		
		clamp_back_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "clamp_back_kernel");
		
		maximum_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "maximum_kernel");
		
		minimum_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "minimum_kernel");
		
		maximum_back_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "maximum_back_kernel");
		
		minimum_back_gpu_function = CUDAModules.getFunctionByModule(LibPaths.LIB_PATH+"OPKernel.cu", "minimum_back_kernel");
		
	}
	
	public static OPKernel getInstance() {
		if(kernel == null) {
			kernel = new OPKernel();
		}
		return kernel;
	}
	
	public void fill_gpu(Tensor x,float val) {
		
		try {

			/**
			 * int N, float ALPHA, float *X
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{x.getDataLength()}),
	                Pointer.to(new float[]{val}),
	                Pointer.to(x.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(fill_gpu_function,
	        		CAFFE_GET_BLOCKS(x.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void copy_gpu(Tensor a,Tensor b,int offset) {
		
		try {

			/**
			 * int N,  float *X, int OFFX, float *Y, int OFFY
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{b.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{offset}),
	        		Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{0})
	            );
			
			checkCUDA(cuLaunchKernel(copy_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void copy_gpu(Tensor a,Tensor b,int offsetX,int offsetY) {
		
		try {

			/**
			 * int N,  float *X, int OFFX, float *Y, int OFFY
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{a.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{offsetX}),
	        		Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{offsetY})
	            );
			
			checkCUDA(cuLaunchKernel(copy_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void copy_number_gpu(Tensor a,Tensor b,int start,int cpy) {
		
		try {

			/**
			 * int N,  float *X, float *Y, int n,int c,int h,int w,int start
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{b.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(new int[]{a.number}),
	        		Pointer.to(new int[]{a.channel}),
	        		Pointer.to(new int[]{a.height}),
	        		Pointer.to(new int[]{a.width}),
	                Pointer.to(new int[]{start}),
	                Pointer.to(new int[]{cpy})
	            );
			
			checkCUDA(cuLaunchKernel(copy_number_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void copy_channel_gpu(Tensor a,Tensor b,int start,int cpy) {
		
		try {

			/**
			 * int N,  float *X, float *Y, int n,int c,int h,int w,int start
			 */
			Pointer kernelParameter = Pointer.to(
					Pointer.to(new int[]{b.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(new int[]{a.number}),
	        		Pointer.to(new int[]{a.channel}),
	        		Pointer.to(new int[]{a.height}),
	        		Pointer.to(new int[]{a.width}),
	                Pointer.to(new int[]{start}),
	                Pointer.to(new int[]{cpy})
	            );
			
			checkCUDA(cuLaunchKernel(copy_channel_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void add_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(add_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void add_gpu(Tensor a,Tensor b,Tensor y,int offset,int N) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{N}),
	                Pointer.to(a.getGpuData().withByteOffset(offset * Sizeof.FLOAT)),
	        		Pointer.to(b.getGpuData().withByteOffset(offset * Sizeof.FLOAT)),
	        		Pointer.to(y.getGpuData().withByteOffset(offset * Sizeof.FLOAT))
	            );
			
			checkCUDA(cuLaunchKernel(add_gpu_function,
	        		CAFFE_GET_BLOCKS(N),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void axpy_gpu(Tensor a,Tensor b,int offsetX,int offsetY) {
		
		try {

			/**
			 * int N,  float *X, int OFFX, float *Y, int OFFY
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{a.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{offsetX}),
	        		Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{offsetY})
	            );
			
			checkCUDA(cuLaunchKernel(axpy_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void add_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(add_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void add_number_gpu(Tensor a,Tensor b,int start) {
		
		try {

			/**
			 * int N,  float *X, float *Y, int n,int c,int h,int w,int start
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{b.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(new int[]{a.number}),
	        		Pointer.to(new int[]{a.channel}),
	        		Pointer.to(new int[]{a.height}),
	        		Pointer.to(new int[]{a.width}),
	                Pointer.to(new int[]{start})
	            );
			
			checkCUDA(cuLaunchKernel(add_number_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void add_channel_gpu(Tensor a,Tensor b,int start) {
		
		try {

			/**
			 * int N,  float *X, float *Y, int n,int c,int h,int w,int start
			 */
			Pointer kernelParameter = Pointer.to(
					Pointer.to(new int[]{b.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(new int[]{a.number}),
	        		Pointer.to(new int[]{a.channel}),
	        		Pointer.to(new int[]{a.height}),
	        		Pointer.to(new int[]{a.width}),
	                Pointer.to(new int[]{start})
	            );
			
			checkCUDA(cuLaunchKernel(add_channel_gpu_function,
	        		CAFFE_GET_BLOCKS(b.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void sub_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(sub_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void sub_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(sub_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void scalar_sub_gpu(float a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	        		Pointer.to(new float[] {a}),
	                Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(scalar_sub_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mul_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(mul_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mul_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {
			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(mul_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mul_plus_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(mul_plus_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mul_plus_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(mul_plus_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void scalar_div_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(scalar_div_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_plus_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_plus_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_plus_scalar_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_plus_scalar_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void scalar_plus_div_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(scalar_plus_div_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_bGrad_gpu(Tensor a,Tensor b,Tensor c,Tensor y) {
		
		try {

			/**
			 * int N, float *A, float *B, float *C, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(c.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_bGrad_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void div_scalar_bGrad_gpu(Tensor a,float b,Tensor c,Tensor y) {
		
		try {

			/**
			 * int N, float *D, float A, float *B, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(c.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(div_scalar_bGrad_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void pow_gpu(Tensor a,float b,Tensor y) {
		
		try {

			/**
			 * int N, float ALPHA, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	        		Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(pow_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void log_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(log_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void exp_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(exp_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void sum_gpu(Tensor a,Tensor y,int axis) {
		
		try {

			if(axis == 0) {
				
				/**
				 * int N, float *X, float *Y
				 */
				Pointer kernelParameter = Pointer.to(
		        		Pointer.to(new int[]{a.dataLength}),
		                Pointer.to(a.getGpuData()),
		        		Pointer.to(y.getGpuData())
		            );
				
				checkCUDA(cuLaunchKernel(sum_gpu_function,
		        		1, 1, 1,      // Grid dimension
			            1, 1, 1,      // Block dimension
			            0, null,               // Shared memory size and stream
			            kernelParameter, null // Kernel- and extra parameters
			        ));
				
			}else {

				/**
				 * int N, float *X, float *Y,int C,int H,int W
				 */
				Pointer kernelParameter = Pointer.to(
		        		Pointer.to(new int[]{y.getDataLength()}),
		                Pointer.to(a.getGpuData()),
		        		Pointer.to(y.getGpuData()),
		        		Pointer.to(new int[]{a.channel}),
		        		Pointer.to(new int[]{a.height}),
		        		Pointer.to(new int[]{a.width})
		            );
				
				checkCUDA(cuLaunchKernel(sum_channel_gpu_function,
		        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
			            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
			            0, null,               // Shared memory size and stream
			            kernelParameter, null // Kernel- and extra parameters
			        ));
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void sin_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(sin_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void cos_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(cos_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void tan_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(tan_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void tan_back_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(tan_back_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void atan_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(atan_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void atan_back_gpu(Tensor a,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float *Y
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(atan_back_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void broadcast_gpu(Tensor a,Tensor y,int axis) {
		
		try {
			
			if(axis == 0){

				/**
				 * int N, float *X, float *Y
				 */
				Pointer kernelParameter = Pointer.to(
		        		Pointer.to(new int[]{y.getDataLength()}),
		                Pointer.to(a.getGpuData()),
		        		Pointer.to(y.getGpuData())
		            );
				
				checkCUDA(cuLaunchKernel(broadcast_gpu_function,
		        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
			            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
			            0, null,               // Shared memory size and stream
			            kernelParameter, null // Kernel- and extra parameters
			        ));
			}else {
				
				/**
				 * int N, float *X, float *Y,int C,int H,int W
				 */
				Pointer kernelParameter = Pointer.to(
		        		Pointer.to(new int[]{y.getDataLength()}),
		                Pointer.to(a.getGpuData()),
		        		Pointer.to(y.getGpuData()),
		        		Pointer.to(new int[]{y.channel}),
		        		Pointer.to(new int[]{y.height}),
		        		Pointer.to(new int[]{y.width})
		            );
				
				checkCUDA(cuLaunchKernel(broadcast_channel_gpu_function,
		        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
			            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
			            0, null,               // Shared memory size and stream
			            kernelParameter, null // Kernel- and extra parameters
			        ));
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void clamp_gpu(Tensor a,float b1,float b2,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b1}),
	        		Pointer.to(new float[] {b2}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(clamp_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void clamp_back_gpu(Tensor a,float b1,float b2,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(new float[] {b1}),
	        		Pointer.to(new float[] {b2}),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(clamp_back_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void maximum_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(maximum_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void minimum_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(minimum_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void maximum_back_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(maximum_back_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void minimum_back_gpu(Tensor a,Tensor b,Tensor y) {
		
		try {

			/**
			 * int N, float *X, float ALPHA, float *R
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{y.getDataLength()}),
	                Pointer.to(a.getGpuData()),
	        		Pointer.to(b.getGpuData()),
	        		Pointer.to(y.getGpuData())
	            );
			
			checkCUDA(cuLaunchKernel(minimum_back_gpu_function,
	        		CAFFE_GET_BLOCKS(y.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void mean_gpu(Tensor a,int dim,Tensor y) {
		
		try {
			
			int scalar = a.number;
			
			if(dim == 1) {
				scalar = a.channel;
			}
			
			sum_gpu(a, y, dim);

			div_scalar_gpu(y, scalar, y);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
}
