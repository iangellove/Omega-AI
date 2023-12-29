package com.omega.engine.ad.op.sign;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.ad.op.gpu.OPKernel;

/**
 * f(a,b) = a / b;
 * da = g / b
 * db = -g * a / b^2
 * @author Administrator
 */
public class DivOP extends SignOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = 6114922229588936622L;
	
	public static DivOP op = null;
	
	public static final OPType opt = OPType.division;
	
	public static DivOP getInstance() {
		if(op == null) {
			op = new DivOP();
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
		if(other != null) {
			TensorOP.div(self, other, y);
		}else {
			TensorOP.div(self, tape.getScalar(), y);
		}
		if(self.isRequiresGrad() || (other != null && other.isRequiresGrad())) {
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
			if(y!=null) {
				TensorOP.divPlus(delta, y, x.getGrad());
			}else {
				TensorOP.divPlus(delta, tape.getScalar(), x.getGrad());
			}
		}
		if(y != null && y.isRequiresGrad()) {
			if(y.getGrad().isHasGPU()) {
				OPKernel.getInstance().div_bGrad_gpu(delta, x, y, y.getGrad());
			}else {
				bGrad(delta.data, x.data, y.data, y.getGrad().data);
			}
		}
		
	}
	
	/**
	 * db = -delta * a / b^2
	 * @param delta
	 * @param a
	 * @param b
	 * @return
	 */
	public static void bGrad(float[] delta,float[] a,float[] b,float[] grad) {
		for(int i = 0;i<delta.length;i++){
			grad[i] += - 1.0f * delta[i] * a[i] / (b[i] * b[i]); 
		}
	}

}
