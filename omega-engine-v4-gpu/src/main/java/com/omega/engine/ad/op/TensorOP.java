package com.omega.engine.ad.op;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.op.gpu.OPKernel;
import com.omega.engine.gpu.GPUOP;

import jcuda.jcublas.cublasOperation;

public class TensorOP {
	
	public static void add(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.add(a.data, b.data);
		}
		
	}
	
	public static void add(Tensor a,Tensor b,Tensor c,int axis) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.add(a.data, b.data);
		}
		
	}
	
	public static void add(Tensor a,Tensor b,Tensor c, int offset,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_gpu(a, b, c, offset, N);
		}else {
			c.data = MatrixOperation.add(a.data, b.data);
		}
		
	}
	
	public static void add(Tensor a,Tensor b,Tensor c, int offsetA,int offsetB,int offsetC,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_gpu(a, b, c, offsetA, offsetB, offsetC, N);
		}else {
			c.data = MatrixOperation.add(a.data, b.data);
		}
		
	}
	
	public static void add(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_scalar_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.add(a.data, b);
		}
		
	}
	
	public static void sub(Tensor a,Tensor b,Tensor c) {
		
		int axis = getAxis(a, b);
		
		if(axis >= 0) {
			sub(a, b, c, axis);
		}else {
			if(c.isHasGPU()) {
				OPKernel.getInstance().sub_gpu(a, b, c);
			}else {
				c.data = MatrixOperation.subtraction(a.data, b.data);
			}
		}
	}
	
	public static int getAxis(Tensor a,Tensor b) {
		if(a.getDataLength() == b.getDataLength()) {
			return -1;
		}
		return 0;
	}
	
	public static void sub(Tensor a,Tensor b,Tensor c,int axis) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().sub_gpu(a, b, c, axis);
		}else {
			c.data = MatrixOperation.subtraction(a.data, b.data, axis);
		}
		
	}
	
	public static void sub(Tensor a,Tensor b,Tensor c,int offset,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().sub_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.subtraction(a.data, b.data);
		}
		
	}
	
	public static void sub(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().sub_scalar_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.subtraction(a.data, b);
		}
		
	}
	
	public static void sub(float a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().scalar_sub_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.subtraction(a, b.data);
		}
		
	}
	
	public static void sub(float a,Tensor b,Tensor c,int offset,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().scalar_sub_gpu(a, b, c, offset, N);
		}else {
			c.data = MatrixOperation.subtraction(a, b.data);
		}
		
	}

	public static void mul(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.multiplication(a.data, b.data);
		}
		
	}
	
	public static void bool(Tensor a,Tensor b,Tensor c,float val) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().bool_gpu(a, b, c, val);
		}else {
			c.data = MatrixOperation.bool(a.data, b.data, val);
		}
		
	}
	
	public static void mul(Tensor a,Tensor b,Tensor c, int offset,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_gpu(a, b, c, offset, N);
		}else {
			c.data = MatrixOperation.multiplication(a.data, b.data);
		}
		
	}
	
	public static void mul(Tensor a,Tensor b,Tensor c, int offsetA,int offsetB,int offsetY,int N) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_gpu(a, b, c, offsetA, offsetB, offsetY, N);
		}else {
			c.data = MatrixOperation.multiplication(a.data, b.data);
		}
		
	}
	
	public static void mul(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_scalar_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.multiplication(a.data, b);
		}
		
	}
	
	public static void mulPlus(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_plus_gpu(a, b, c);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b.data));
		}
		
	}
	
	public static void mulPlus(Tensor a,float b,Tensor c) {
		
		int axis = getAxis(a, c);
		
		if(axis >= 0) {
			mulPlus(a, b, c, axis);
		}else {

			if(c.isHasGPU()) {
				OPKernel.getInstance().mul_plus_scalar_gpu(a, b, c);
			}else {
				MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b));
			}
		
		}
		
	}
	
	public static void mulPlus(Tensor a,float b,Tensor c,int axis) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_plus_scalar_gpu(a, b, c, axis);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b), axis);
		}
		
	}
	
	public static void div(Tensor a,Tensor b,Tensor c) {
		int axis = getAxis(a, b);
		if(axis >= 0) {
			div(a, b, c, axis);
		}else {
			if(c.isHasGPU()) {
				OPKernel.getInstance().div_gpu(a, b, c);
			}else {
				c.data = MatrixOperation.division(a.data, b.data);
			}
		}
	}
	
	public static void div(Tensor a,Tensor b,Tensor c,int axis) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_gpu(a, b, c, axis);
		}else {
			c.data = MatrixOperation.division(a.data, b.data, axis);
		}
		
	}
	
	public static void div(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_scalar_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.division(a.data, b);
		}
		
	}
	
	public static void div(float a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().scalar_div_gpu(b, a, c);
		}else {
			c.data = MatrixOperation.division(a, b.data);
		}
		
	}
	
	public static void divPlus(Tensor a,Tensor b,Tensor c) {
		
		int axis = getAxis(a, b);

		if(axis >= 0) {
			
			divPlus(a, b, c, axis);
		}else {
			if(c.isHasGPU()) {
				OPKernel.getInstance().div_plus_gpu(a, b, c);
			}else {
				MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b.data));
			}
		}
		
	}
	
	public static void divPlus(Tensor a,Tensor b,Tensor c,int axis) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_plus_gpu(a, b, c, axis);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b.data, axis));
		}
		
	}
	
	public static void divPlus(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_plus_scalar_gpu(a, b, c);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b));
		}
		
	}
	
	public static void exp(Tensor a,Tensor b) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().exp_gpu(a, b);
		}else {
			b.data = MatrixOperation.exp(a.data);
		}
		
	}
	
	public static void transpose(Tensor a,Tensor b) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().transpose_gpu(a, b);
		}else {
//			b.data = MatrixOperation.exp(a.data);
		}
		
	}
	
	public static void sum(Tensor a,Tensor b,int axis) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().sum_gpu(a, b, axis);
		}else {
			b.data = MatrixOperation.sum(a.data, a.number, a.channel, a.height, a.width, axis);
		}
		
	}
	
	public static void max(Tensor a,Tensor b,int axis) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().max_gpu(a, b, axis);
		}else {
			b.data = MatrixOperation.max(a.data, a.number, a.channel, a.height, a.width, axis);
		}
		
	}
	
	public static void max_backward(Tensor d,Tensor a,Tensor b,int axis) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().max_backward_gpu(d, a, b, axis);
		}else {
			b.data = MatrixOperation.max(a.data, a.number, a.channel, a.height, a.width, axis);
		}
		
	}
	
	public static void log(Tensor a,Tensor b) {
		
		if(b.isHasGPU()) {
			OPKernel.getInstance().log_gpu(a, b);
		}else {
			b.data = MatrixOperation.log(a.data);
		}
		
	}
	
	public static void pow(Tensor a,float b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().pow_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.pow(a.data, b);
		}
		
	}
	
	public static void sqrt(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().sqrt_gpu(a, c);
		}else {
			c.data = MatrixOperation.sqrt(a.data);
		}
		
	}
	
	public static void sin(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().sin_gpu(a, c);
		}else {
			c.data = MatrixOperation.sin(a.data);
		}
		
	}
	
	public static void cos(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().cos_gpu(a, c);
		}else {
			c.data = MatrixOperation.cos(a.data);
		}
		
	}
	
	public static void tan(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().tan_gpu(a, c);
		}else {
			c.data = MatrixOperation.tan(a.data);
		}
		
	}
	
	public static void tan_back(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().tan_back_gpu(a, c);
		}else {
			c.data = MatrixOperation.tan_back(a.data);
		}
		
	}
	
	public static void atan(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().atan_gpu(a, c);
		}else {
			c.data = MatrixOperation.atan(a.data);
		}
		
	}
	
	public static void atan_back(Tensor a,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().atan_back_gpu(a, c);
		}else {
			c.data = MatrixOperation.atan_back(a.data);
		}
		
	}
	
	public static void broadcast(Tensor a,Tensor c,int axis) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().broadcast_plus_gpu(a, c, axis);
		}else {
			MatrixOperation.broadcast_plus(a.data, c.data, c.number, c.channel, c.height, c.width, axis);
		}
	}
	
	public static void clamp(Tensor a,float b1,float b2,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().clamp_gpu(a, b1, b2, c);
		}else {
			c.data = MatrixOperation.clamp(a.data, b1, b2);
		}
	}
	
	public static void clamp_back(Tensor a,float b1,float b2,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().clamp_back_gpu(a, b1, b2, c);
		}else {
			c.data = MatrixOperation.clamp_back(a.data, b1, b2);
		}
	}
	
	public static void maximum(Tensor a,Tensor b,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().maximum_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.maximum(a.data, b.data);
		}
	}
	
	public static void minimum(Tensor a,Tensor b,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().minimum_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.minimum(a.data, b.data);
		}
	}
	
	public static void maximum_back(Tensor a,Tensor b,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().maximum_back_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.maximum_back(a.data, b.data);
		}
	}
	
	public static void minimum_back(Tensor a,Tensor b,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().minimum_back_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.minimum_back(a.data, b.data);
		}
	}
	
	public static void mean(Tensor a,int dim,Tensor c) {
		if(c.isHasGPU()) {
			OPKernel.getInstance().mean_gpu(a, dim, c);
		}else {
			c.data = MatrixOperation.mean(a.data, a.number, a.channel, a.height, a.width, dim);
		}
	}
	
	/**
	 * [M,N] dot [N,K]
	 * @param a
	 * @param b
	 * @param c
	 * @param A_OP
	 * @param b_OP
	 */
	public static void dot(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			/**
			 * m = M,n = K,k = N
			 * batch, oWidth, width
			 */
//			System.out.println(JsonUtils.toJson(a.shape()));
//			System.out.println(JsonUtils.toJson(b.shape()));
//			a.showDM();
//			b.showDM();
			int k = b.number;
			if(b.number == 1) {
				k = b.height;
			}
			GPUOP.getInstance().multiplyFloat(a.number, b.width, k, a.getGpuData(), b.getGpuData(), c.getGpuData(),
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
//			c.showDM();
//			System.out.println("----------------------");
		}else {
//			c.data = MatrixOperation.dot(a.data, b.data);
		}
		
	}

	/**
	 * diff = delta * weightT
	 * this.number, this.width, this.oWidth
	 * @param a
	 * @param b
	 * @param c
	 */
	public static void dotDX(Tensor a,Tensor b,Tensor c) {
		
		int k = b.number;
		if(b.number == 1) {
			k = b.height;
		}
		GPUOP.getInstance().multiplyFloat(a.number, k, b.width, a.getGpuData(), b.getGpuData(), c.getGpuData(),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 1.0f);
		
	}
	
	/**
	 * deltaW = inputT * delta
	 * this.width, this.oWidth, this.number
	 * @param a
	 * @param b
	 * @param c
	 */
	public static void dotDW(Tensor a,Tensor b,Tensor c) {
		
		GPUOP.getInstance().multiplyFloat(a.width, b.width, a.number, a.getGpuData(), b.getGpuData(), c.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 1.0f);
		
	}
	
	public static void permute(Tensor a,Tensor b,int[] permutes) {
		
		if(a.isHasGPU()) {
			OPKernel.getInstance().permute_gpu(a, b, permutes);
		}else {
//			c.data = MatrixOperation.add(a.data, b.data);
		}
		
	}
	
}
