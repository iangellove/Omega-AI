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

public class SinOP extends SignOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -7252060328891832266L;
	
	public static SinOP op = null;
	
	public static final OPType opt = OPType.sin;
	
	public static SinOP getInstance() {
		if(op == null) {
			op = new SinOP();
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tensor self, Tensor other) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.sin(self.data));
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		List<Tensor> inputs = new ArrayList<Tensor>(1);
		inputs.add(self);
		List<Tensor> outputs = new ArrayList<Tensor>(1);
		outputs.add(y);
		Tape tape = new Tape(inputs, outputs, this);
		Graph.add(tape);
		return y;
	}

	@Override
	public void backward(float[] delta, List<Tensor> inputs) {
		// TODO Auto-generated method stub
		System.out.println("sin-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			float[] dy_dself = MatrixOperation.cos(inputs.get(0).data);
			if(inputs.get(0).getGrad() != null) {
				inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), MatrixOperation.multiplication(delta, dy_dself)));
			}else {
				inputs.get(0).setGrad(MatrixOperation.multiplication(delta, dy_dself));
			}
			System.out.println("sin--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
		}
	}

}
