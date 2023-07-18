package com.omega.engine.ad.op.sign;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.SignOP;


/**
 * f(scalar,b) = scalar / b;
 * db = -g * scalar / b^2
 * @author Administrator
 */
public class ScalarDivOP extends SignOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3087002822041265440L;

	public static ScalarDivOP op = null;
	
	public static final OPType opt = OPType.scalarDivision;
	
	public static ScalarDivOP getInstance() {
		if(op == null) {
			op = new ScalarDivOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tensor self, Tensor other) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor forward(Tensor self, float scalar) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.division(scalar, self.data));
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		List<Tensor> inputs = new ArrayList<Tensor>(1);
		inputs.add(self);
		List<Tensor> outputs = new ArrayList<Tensor>(1);
		outputs.add(y);
		Tape tape = new Tape(inputs, outputs, this, scalar);
		Graph.add(tape);
		return y;
	}

	@Override
	public void backward(float[] delta, List<Tensor> inputs, float scalar) {
		// TODO Auto-generated method stub
		System.out.println("div-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			if(inputs.get(0).getGrad() != null) {
				inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), bGrad(delta, scalar, inputs.get(0).data)));
			}else {
				inputs.get(0).setGrad(bGrad(delta, scalar, inputs.get(0).data));
			}
		}
	}
	
	/**
	 * db = -delta * scalar / b^2
	 * @param delta
	 * @param scalar
	 * @param b
	 * @return
	 */
	public static float[] bGrad(float[] delta,float scalar,float[] b) {
		float[] grad = new float[delta.length];
		for(int i = 0;i<delta.length;i++){
			grad[i] = - 1.0f * delta[i] * scalar / (b[i] * b[i]); 
		}
		return grad;
	}
	
}
