package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;

/**
 * ConvBaseKernel
 * @author Administrator
 *
 */
public abstract class FullyBaseKernel extends BaseKernel{
	
	public abstract void conv(Tensor input,Tensor kernel,Tensor output);
	
	public abstract void convTranspose(Tensor input,Tensor kernel,Tensor output);
	
	public abstract void dw(Tensor input,Tensor delta,Tensor diffW);
	
	public abstract void dx(Tensor delta,Tensor kernel,Tensor diff);
	
}
