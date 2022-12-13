package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;

public abstract class LossFunction {
	
	public LossType lossType;
	
	public float eta = 0.00001f;
	
	public abstract Tensor loss(Tensor x, Tensor label);
	
	public abstract Tensor diff(Tensor x, Tensor label);
	
	public abstract LossType getLossType();
	
	public float gradientCheck(Tensor x, Tensor label) {
		Tensor f1 = this.loss(new Tensor(x.number, x.channel, x.height, x.width, MatrixOperation.add(x.data, eta), true), label);
		Tensor f2 = this.loss(new Tensor(x.number, x.channel, x.height, x.width, MatrixOperation.subtraction(x.data, eta), true), label);
		Tensor diff = this.diff(x,label);
		float[] temp = MatrixOperation.subtraction(f1.data, f2.data);
		temp = MatrixOperation.division(temp, 2 * eta);
		System.out.println("diff:"+JsonUtils.toJson(diff.syncHost()));
		System.out.println("gradientCheck:"+JsonUtils.toJson(temp));
		float[] error = MatrixOperation.subtraction(diff.data, temp);
		return MatrixOperation.sum(error);
	} 
	
}
