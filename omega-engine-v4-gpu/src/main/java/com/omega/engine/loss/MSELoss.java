package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.data.Tensors;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Graph;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class MSELoss extends LossFunction {
	
	public final LossType lossType = LossType.MSE;
	
	private static MSELoss instance;
	
	public static MSELoss operation() {
		if(instance == null) {
			instance = new MSELoss();
		}
		return instance;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.MSE;
	}
	
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		x.setRequiresGrad(true);
		Graph.start();
		Tensor loss = label.sub(x).pow(2.0f).div(2.0f).sum(0).div(x.number);
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		Graph.clearGrad();
		Graph.backward();
		return x.getGrad();
	}
	
	public static void main(String[] args) {
		float[] x = new float[] {0.2f,0.3f,0.5f,0.1f,0.1f,0.8f,0.3f,0.1f,0.6f,0.9f,0.01f,0.09f};
		Tensor xt = Tensors.tensor(4, 1, 1, 3, x);
		float[] label = new float[] {0,1,0,1,0,0,1,0,0,0,0,1};
		Tensor labelt = Tensors.tensor(4, 1, 1, 3, label);
		float error = MSELoss.operation().gradientCheck(xt,labelt);
		System.out.println("error:"+error);
	}

	@Override
	public Tensor[] loss(Tensor[] x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor[] diff(Tensor[] x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

}
