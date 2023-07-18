package com.omega.engine.ad.op;

import com.omega.common.data.Tensor;

public abstract class FunctionOP extends OP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -4251342408266424838L;

	public abstract Tensor forward(Tensor self);
	
}
