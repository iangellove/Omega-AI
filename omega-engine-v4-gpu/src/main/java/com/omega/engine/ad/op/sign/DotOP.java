package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;

public class DotOP extends SignOP {


	/**
	 * 
	 */
	private static final long serialVersionUID = 8033645023767349745L;
	
	public static DotOP op = null;
	
	public static final OPType opt = OPType.dot;
	
	public static DotOP getInstance() {
		if(op == null) {
			op = new DotOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub

	}

}
