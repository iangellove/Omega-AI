package com.omega.engine.ad.op.functions;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;

/**
 * ATanOP
 * @author Administrator
 *
 */
public class ATanOP extends FunctionOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7252060328891832266L;
	
	public static ATanOP op = null;
	
	public static final OPType opt = OPType.atan;
	
	public static ATanOP getInstance() {
		if(op == null) {
			op = new ATanOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		Tensor self = tape.getX();
		Tensor y = tape.getOutput();
		TensorOP.atan(self, y);
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
			/**
			 * dx = 1 / (1 + x^2) 
			 */
			Tensor dy = tape.getTmp();
			TensorOP.atan_back(x, dy);
			TensorOP.mulPlus(delta, dy, x.getGrad());
		}
	}

}