package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.data.Tensors;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class SquareLoss extends LossFunction {
	
	public final LossType lossType = LossType.square_loss;
	
	private static SquareLoss instance;
	
	public static SquareLoss operation() {
		if(instance == null) {
			instance = new SquareLoss();
		}
		return instance;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.square_loss;
	}
	
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		Tensor temp = Tensors.tensor(x.number,x.channel,x.height,x.width);
		
		for(int i = 0;i<x.getDataLength();i++) {
			temp.data[i] = (x.data[i] - label.data[i]) * (x.data[i] - label.data[i]) / 2;;
		}
		
		return temp;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		Tensor temp = Tensors.tensor(x.number,x.channel,x.height,x.width);
		
		for(int i = 0;i<x.getDataLength();i++) {
			temp.data[i] = x.data[i] - label.data[i];
		}
		
		
		return temp;
	}
	
	public static void main(String[] args) {
		float[] x = new float[] {0.2f,0.3f,0.5f,0.1f,0.1f,0.8f,0.3f,0.1f,0.6f,0.9f,0.01f,0.09f};
		Tensor xt = Tensors.tensor(4, 1, 1, 3, x);
		float[] label = new float[] {0,1,0,1,0,0,1,0,0,0,0,1};
		Tensor labelt = Tensors.tensor(4, 1, 1, 3, label);
		float error = SquareLoss.operation().gradientCheck(xt,labelt);
		System.out.println("error:"+error);
	}

}
