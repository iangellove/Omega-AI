package com.omega.engine.ad.op.functions;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;

public class PowOP extends FunctionOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3857343378511617891L;

	public static PowOP op = null;
	
	public static final OPType opt = OPType.pow;
	
	public static PowOP getInstance() {
		if(op == null) {
			op = new PowOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		Tensor self = tape.getX();
		Tensor y = tape.getOutput();
		TensorOP.pow(self, tape.getScalar(), y);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		return y;
	}
	
	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor x = tape.getX();
		if(x.isRequiresGrad()) {
			Tensor dy = tape.getTmp();
			TensorOP.mul(x, tape.getScalar(), dy);
			TensorOP.pow(dy, tape.getScalar() - 1, dy);
			TensorOP.mulPlus(delta, dy, x.getGrad());
		}
	}

}
