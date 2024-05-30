package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;

public abstract class PoolingBaseKernel extends BaseKernel{
	
	public abstract void forward(Tensor input, Tensor output);
	
	public abstract void backward(Tensor input,Tensor output,Tensor delta,Tensor diff);
	
}
