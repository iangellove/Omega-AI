package com.omega.engine.ad.op;

import java.io.Serializable;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;

public abstract class OP implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 774715895241216806L;
	
	private OPType opType;
	
	public abstract Tensor forward(Tape tape);
	
	public abstract void backward(Tensor delta,Tape tape);
	
	public OPType getOpType() {
		return opType;
	}

	public void setOpType(OPType opType) {
		this.opType = opType;
	}

}
