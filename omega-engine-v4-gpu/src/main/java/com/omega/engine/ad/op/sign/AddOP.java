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
 * 加法操作
 * @author Administrator
 *
 */
public class AddOP extends SignOP{

	/**
	 * 
	 */
	private static final long serialVersionUID = -6030727723343775529L;
	
	public static AddOP op = null;
	
	public static final OPType opt = OPType.add;
	
	public static AddOP getInstance() {
		if(op == null) {
			op = new AddOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tensor self, Tensor other) {
		// TODO Auto-generated method stub
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.add(self.data, other.data));
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
		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.add(self.data, scalar));
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

//	@Override
//	public Tensor forward(Tensor self, Tensor other, int[] position) {
//		// TODO Auto-generated method stub
//		int n = self.getNumber();
//		int c = self.getChannel();
//		int h = self.getHeight();
//		int w = self.getWidth();
//		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.addToY(self.data, other.data, n, c, h, w, position));
//		if(self.isRequiresGrad() || other.isRequiresGrad()) {
//			y.setRequiresGrad(true);
//		}
//		List<Tensor> inputs = new ArrayList<Tensor>(2);
//		inputs.add(self);
//		inputs.add(other);
//		List<Tensor> outputs = new ArrayList<Tensor>(1);
//		outputs.add(y);
//		Tape tape = new Tape(inputs, outputs, this, position);
//		Graph.add(tape);
//		return y;
//	}

//	@Override
//	public Tensor forward(Tensor self, float scalar, int[] position) {
//		// TODO Auto-generated method stub
//		int n = self.getNumber();
//		int c = self.getChannel();
//		int h = self.getHeight();
//		int w = self.getWidth();
//		Tensor y = new Tensor(self.number, self.channel, self.height, self.width, MatrixOperation.addToY(self.data, scalar, n, c, h, w, position));
//		if(self.isRequiresGrad()) {
//			y.setRequiresGrad(true);
//		}
//		List<Tensor> inputs = new ArrayList<Tensor>(1);
//		inputs.add(self);
//		List<Tensor> outputs = new ArrayList<Tensor>(1);
//		outputs.add(y);
//		Tape tape = new Tape(inputs, outputs, this, scalar, position);
//		Graph.add(tape);
//		return y;
//	}

	@Override
	public void backward(float[] delta,List<Tensor> inputs,float scalar) {
		// TODO Auto-generated method stub
		System.out.println("add-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			if(inputs.get(0).getGrad() != null) {
				inputs.get(0).setGrad(MatrixOperation.add(inputs.get(0).getGrad(), delta));
			}else {
				inputs.get(0).setGrad(delta);
			}
		}
		System.out.println("add--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
		if(inputs.size() > 1 && inputs.get(1).isRequiresGrad()) {
			if(inputs.get(1).getGrad() != null) {
				inputs.get(1).setGrad(MatrixOperation.add(inputs.get(1).getGrad(), delta));
			}else {
				inputs.get(1).setGrad(delta);
			}
			System.out.println("add--d2:"+JsonUtils.toJson(inputs.get(1).getGrad()));
		}
	}

}
