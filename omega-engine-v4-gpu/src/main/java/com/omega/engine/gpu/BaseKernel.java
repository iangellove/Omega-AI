package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

public class BaseKernel {
	
	public int N = 0;
	
	private int CAFFE_CUDA_NUM_THREADS = 1024;
	
	private CUfunction copy_gpu_function;
	
	private CUfunction axpy_gpu_function;
	
	private CUfunction fill_gpu_function;
	
	public BaseKernel() {
		
		copy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "copy_kernel");
		
		axpy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "axpy_kernel");
		
		fill_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "fill_kernel");
		
	}
	
	public void fill_gpu(Tensor a,float val) {
		
		try {

			if(fill_gpu_function == null) {
				fill_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "fill_kernel");
			}
			
			/**
			 * int N, float ALPHA, float *X, int INCX
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{a.getDataLength()}),
	                Pointer.to(new float[]{val}),
	        		Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{1})
	            );
			
			checkCUDA(cuLaunchKernel(fill_gpu_function,
	        		CAFFE_GET_BLOCKS(a.getDataLength()),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        ));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void axpy_gpu(Tensor a,Tensor b,int N, float ALPHA,int INCX, int INCY) {
		axpy_gpu(a, b, INCY, ALPHA, 0, INCX, 0, INCY);
	}
	
	public void axpy_gpu(Tensor a,Tensor b,int N, float ALPHA, int OFFX, int INCX, int OFFY, int INCY) {
		
		try {
			
			if(axpy_gpu_function == null) {
				axpy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "axpy_kernel");
			}
			
			/**
			 * int N, float ALPHA, float *X, int OFFX, int INCX,  float *Y, int OFFY, int INCY
			 */
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{N}),
	                Pointer.to(new float[]{ALPHA}),
	        		Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{OFFX}),
	                Pointer.to(new int[]{INCX}),
	                Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{OFFY}),
	                Pointer.to(new int[] {INCY})
	            );
			
			checkCUDA(cuLaunchKernel(axpy_gpu_function,
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

	public void copy_gpu(Pointer a,Pointer b,int N,int incx,int incy) {
		
		try {
			
			if(copy_gpu_function == null) {
				copy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "copy_kernel");
			}
			
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(a),
	                Pointer.to(new int[]{0}),
	                Pointer.to(new int[]{incx}),
	                Pointer.to(b),
	                Pointer.to(new int[]{0}),
	                Pointer.to(new int[] {incy})
	            );
			
			checkCUDA(cuLaunchKernel(copy_gpu_function,
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
	
	public void copy_gpu(Tensor a,Tensor b,int N,int incx,int incy) {
		
		try {
			
			if(copy_gpu_function == null) {
				copy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "copy_kernel");
			}
			
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{0}),
	                Pointer.to(new int[]{incx}),
	                Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{0}),
	                Pointer.to(new int[] {incy})
	            );
			
	        cuLaunchKernel(copy_gpu_function,
	        		CAFFE_GET_BLOCKS(N),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void copy_gpu(Tensor a,Tensor b,int N,int offx,int incx,int offy,int incy) {
		
		try {
			
			if(copy_gpu_function == null) {
				copy_gpu_function = CUDAModules.getFunctionByModule("H://BaseKernel.cu", "copy_kernel");
			}
			
			Pointer kernelParameter = Pointer.to(
	        		Pointer.to(new int[]{N}),
	        		Pointer.to(a.getGpuData()),
	                Pointer.to(new int[]{offx}),
	                Pointer.to(new int[]{incx}),
	                Pointer.to(b.getGpuData()),
	                Pointer.to(new int[]{offy}),
	                Pointer.to(new int[] {incy})
	            );
			
	        cuLaunchKernel(copy_gpu_function,
	        		CAFFE_GET_BLOCKS(N),  1, 1,      // Grid dimension
		            CAFFE_CUDA_NUM_THREADS, 1, 1,      // Block dimension
		            0, null,               // Shared memory size and stream
		            kernelParameter, null // Kernel- and extra parameters
		        );

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public int CAFFE_GET_BLOCKS(int N) {
	    return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
	}
	
	public void showDM(Pointer p,float[] data) {
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void showDM(Pointer p,int[] data) {
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void showDM(Pointer p,int size) {
		float[] data = new float[size];
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
}
