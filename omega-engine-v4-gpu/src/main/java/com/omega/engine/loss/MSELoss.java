package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.data.Tensors;
import com.omega.engine.loss.gpu.MSELossKernel;

/**
 * Square loss
 * @author Administrator
 * @loss: ∑ (y - f(x))^2
 * @diff: 2 * (y - f(x))
 */
public class MSELoss extends LossFunction {
	
	public final LossType lossType = LossType.MSE;
	
	private static MSELoss instance;
	
	private MSELossKernel kernel;
	
	private Tensor loss;
	
	private Tensor diff;
	
	public static MSELoss operation() {
		if(instance == null) {
			instance = new MSELoss();
		}
		return instance;
	}
	
	public MSELoss() {
		kernel = new MSELossKernel();
	}
	
	public void init(Tensor input) {
		if(loss == null || loss.number != input.number) {
			this.loss = new Tensor(input.number, 1, 1, 1, true);
//			this.output = new Tensor(input.number, input.channel, input.height, input.width, true);
			this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
		}
	}
	
//	public static MSELoss operation() {
//		if(instance == null) {
//			instance = new MSELoss();
//		}
//		return instance;
//	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.MSE;
	}
	
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		init(x);
//		x.showDM();
		kernel.forward(x, label, loss);
//		loss.showDM();
//		x.setRequiresGrad(true);
//		Graph.start();
//		Tensor loss = label.sub(x).pow(2.0f).div(2.0f).sum(0).div(x.number);
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		kernel.backward(x, label, diff);
		return diff;
//		Graph.clearGrad();
//		Graph.backward();
//		return x.getGrad();
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

	
	public static void main(String[] args) {
		float[] x = new float[] {0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f,0.7f,0.8f,0.9f};
		Tensor xt = Tensors.tensor(2, 1, 1, 9, x, true);
		float[] label = new float[] {0,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0};
		Tensor labelt = Tensors.tensor(2, 1, 1, 9, label, true);
		Tensor loss = MSELoss.operation().loss(xt, labelt);
		
		loss.showDM();
		Tensor diff = MSELoss.operation().diff(xt, labelt);
		diff.showDM();

	}

	@Override
	public Tensor loss(Tensor x, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		init(x);
		kernel.forward(x, label, loss);
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		kernel.backward(x, label, diff);
		return diff;
	}
}
