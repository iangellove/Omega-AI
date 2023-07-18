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
	public Tensor forward(Tensor self, Tensor other) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.division(self.data, other.data));
		if(self.isRequiresGrad() || other.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		List<Tensor> inputs = new ArrayList<Tensor>(2);
		inputs.add(self);
		inputs.add(other);
		List<Tensor> outputs = new ArrayList<Tensor>(1);
		outputs.add(y);
		Tape tape = new Tape(inputs, outputs, this);
		Graph.add(tape);
		return y;
	}

	@Override
	public Tensor forward(Tensor self, float scalar) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.division(self.data, scalar));
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
	public void backward(float[] delta, List<Tensor> inputs,float scalar) {
		// TODO Auto-generated method stub
		System.out.println("div-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			if(inputs.size() > 1) {
				if(inputs.get(0).getGrad() != null) {
					inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), MatrixOperation.division(delta, inputs.get(1).data)));
				}else {
					inputs.get(0).setGrad(MatrixOperation.division(delta, inputs.get(1).data));
				}
			}else {
				if(inputs.get(0).getGrad() != null) {
					inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), MatrixOperation.division(delta, scalar)));
				}else {
					inputs.get(0).setGrad(MatrixOperation.division(delta, scalar));
				}
			}
		}
		System.out.println("div--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
		if(inputs.size() > 1 && inputs.get(1).isRequiresGrad()) {
			if(inputs.get(1).getGrad() != null) {
				inputs.get(1).setGrad(MatrixOperation.add(inputs.get(1).getGrad(), bGrad(delta, inputs.get(0).data, inputs.get(1).data)));
			}else {
				inputs.get(1).setGrad(bGrad(delta, inputs.get(0).data, inputs.get(1).data));
			}
			System.out.println("div--d2:"+JsonUtils.toJson(inputs.get(1).getGrad()));
		}
		
	}
	
	/**
	 * db = -delta * a / b^2
	 * @param delta
	 * @param a
	 * @param b
	 * @return
	 */
	public static float[] bGrad(float[] delta,float[] a,float[] b) {
		float[] grad = new float[delta.length];
		for(int i = 0;i<delta.length;i++){
			grad[i] = - 1.0f * delta[i] * a[i] / (b[i] * b[i]); 
		}
		return grad;
	}

}
