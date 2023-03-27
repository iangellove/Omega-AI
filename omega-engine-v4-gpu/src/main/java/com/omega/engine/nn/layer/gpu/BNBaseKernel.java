package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.network.RunModel;

public abstract class BNBaseKernel extends BaseKernel{
	
	public abstract void forward(RunModel RUN_MODEL, Tensor gama, Tensor beta, Tensor input, Tensor output);
	
	public abstract void backward(Tensor input,Tensor delta,Tensor diff,Tensor gama,Tensor dgama,Tensor dbeta);
	
}
