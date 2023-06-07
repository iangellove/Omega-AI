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

public class MulOP extends SignOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -4667315516225564503L;
	
	public static MulOP op = null;
	
	public static final OPType opt = OPType.multiplication;
	
	public static MulOP getInstance() {
		if(op == null) {
			op = new MulOP();
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tensor self, Tensor other) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.multiplication(self.data, other.data));
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
	public void backward(float[] delta, List<Tensor> inputs) {
		// TODO Auto-generated method stub
		System.out.println("mul-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			float[] dy_dself = inputs.get(1).data;
			if(inputs.get(0).getGrad() != null) {
				inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), MatrixOperation.multiplication(delta, dy_dself)));
			}else {
				inputs.get(0).setGrad(MatrixOperation.multiplication(delta, dy_dself));
			}
		}
		System.out.println("mul--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
		if(inputs.get(1).isRequiresGrad()) {
			float[] dy_dother = inputs.get(0).data;
			if(inputs.get(1).getGrad() != null) {
				inputs.get(1).setGrad(MatrixOperation.add(inputs.get(1).getGrad(), MatrixOperation.multiplication(delta, dy_dother)));
			}else {
				inputs.get(1).setGrad(MatrixOperation.multiplication(delta, dy_dother));
			}
		}
		System.out.println("mul--d2:"+JsonUtils.toJson(inputs.get(1).getGrad()));
	}

}
