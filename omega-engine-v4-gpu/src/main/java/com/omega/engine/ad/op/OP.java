package com.omega.engine.ad.op;

import java.io.Serializable;
import java.util.List;

import com.omega.common.data.Tensor;

public abstract class OP implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 774715895241216806L;
	
	private OPType opType;
	
	public abstract void backward(float[] delta,List<Tensor> inputs);
	
	public OPType getOpType() {
		return opType;
	}

	public void setOpType(OPType opType) {
		this.opType = opType;
	}

}
