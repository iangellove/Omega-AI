package com.omega.engine.ad.op.functions;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;

/**
 * MaximumOP
 * @author Administrator
 *
 */
public class MaximumOP extends FunctionOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6072156179108651118L;

	public static MaximumOP op = null;
	
	public static final OPType opt = OPType.maximum;
	
	public static MaximumOP getInstance() {
		if(op == null) {
			op = new MaximumOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		Tensor self = tape.getX();
		Tensor other = tape.getY();
		Tensor y = tape.getOutput();
		TensorOP.maximum(self, other, y);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		return y;
	}

	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor x = tape.getX();
		Tensor y = tape.getY();
		if(x.isRequiresGrad()) {
			Tensor dy = tape.getTmp();
			TensorOP.maximum_back(x, y, dy);
			TensorOP.mulPlus(delta, dy, x.getGrad());
		}
		if(y != null && y.isRequiresGrad()) {
			Tensor dy = tape.getTmp();
			TensorOP.minimum_back(x, y, dy);
			TensorOP.mulPlus(delta, dy, y.getGrad());
		}
	}

}
