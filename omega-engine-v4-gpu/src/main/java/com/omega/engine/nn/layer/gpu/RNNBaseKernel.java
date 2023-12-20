package com.omega.engine.nn.layer.gpu;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;

public abstract class RNNBaseKernel extends BaseKernel{
	
	public int seqLength = 1;
	
	public abstract void init(int number);
	
	public abstract long weightSize();
	
	public abstract void forward(Tensor input,Tensor weight,Tensor output);
	
	public abstract void dw(Tensor delta,Tensor output,Tensor input,Tensor dw);
	
	public abstract void dx(Tensor delta,Tensor output,Tensor weight,Tensor diff);
	
	public abstract void initWeights(Tensor w);
	
}
