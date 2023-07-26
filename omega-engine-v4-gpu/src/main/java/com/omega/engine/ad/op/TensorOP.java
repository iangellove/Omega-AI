package com.omega.engine.ad.op;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.op.gpu.OPKernel;

public class TensorOP {
	
	public static void add(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().add_gpu(a, b, c);
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

	public static void mul(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_gpu(a, b, c);
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
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().mul_plus_scalar_gpu(a, b, c);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.multiplication(a.data, b));
		}
		
	}
	
	public static void div(Tensor a,Tensor b,Tensor c) {
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_gpu(a, b, c);
		}else {
			c.data = MatrixOperation.division(a.data, b.data);
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
		
		if(c.isHasGPU()) {
			OPKernel.getInstance().div_plus_gpu(a, b, c);
		}else {
			MatrixOperation.plus(c.data, MatrixOperation.division(a.data, b.data));
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
	
}
