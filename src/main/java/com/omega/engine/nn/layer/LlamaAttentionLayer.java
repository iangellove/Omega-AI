package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;

public abstract class LlamaAttentionLayer extends Layer{
	
	public abstract void forward(Tensor cos,Tensor sin,Tensor input);
	
	public abstract void back(Tensor cos,Tensor sin,Tensor delta);
	
}
