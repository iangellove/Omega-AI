package com.omega.engine.ad.op.functions;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;

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
	public Tensor forward(Tensor self) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.pow(self.data, 2));
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
	public void backward(float[] delta, List<Tensor> inputs,float scalar) {
		// TODO Auto-generated method stub
		if(inputs.get(0).isRequiresGrad()) {
			float[] dy_dself = MatrixOperation.multiplication(inputs.get(0).data, 2);
			if(inputs.get(0).getGrad() != null) {
				inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), MatrixOperation.multiplication(delta, dy_dself)));
			}else {
				inputs.get(0).setGrad(MatrixOperation.multiplication(delta, dy_dself));
			}
			System.out.println("pow--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
		}
	}

}
