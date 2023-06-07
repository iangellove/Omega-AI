package com.omega.engine.ad.op;

import com.omega.common.data.Tensor;

public abstract class SignOP extends OP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -797197687458377304L;
	
	public abstract Tensor forward(Tensor self, Tensor other);
	
}
