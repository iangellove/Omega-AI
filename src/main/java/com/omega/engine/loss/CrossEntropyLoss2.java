package com.omega.engine.loss;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.loss.gpu.CrossEntropyKernel;

/**
 * Cross Entropy loss function
 * 
 * @author Administrator
 *
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 *
 */
public class CrossEntropyLoss2 extends LossFunction {

	public final LossType lossType = LossType.softmax_with_cross_entropy;
	
	private static CrossEntropyLoss2 instance;
	
//	private Tensor output;
	
	private Tensor loss;
	
	private Tensor diff;
	
//	private SoftmaxKernel softmaxKernel;
	
	private CrossEntropyKernel crossEntropyKernel;
	
	public CrossEntropyLoss2() {
		initKernel();
	}
	
	public static CrossEntropyLoss2 operation() {
		if(instance == null) {
			instance = new CrossEntropyLoss2();
		}
		return instance;
	}
	
	public void init(Tensor input) {
		if(loss == null || loss.number != input.number) {
			this.loss = new Tensor(input.number, 1, 1, 1, true);
//			this.output = new Tensor(input.number, input.channel, input.height, input.width, true);
			this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
		}
	}
	
	public void initKernel() {
//		softmaxKernel = new SoftmaxKernel();
		crossEntropyKernel = new CrossEntropyKernel();
	}
	
	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.cross_entropy;
	}

	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		
		init(x);
		
		/**
		 * q(x) = softmax(x)
		 * H(p,q) = - ∑p(x)logq(x)
		 * 简化log_softmax:
		 * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
		 * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
		 */
		crossEntropyKernel.forward(x, label, loss);
		
		return loss;
	}
	
	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		
		/**
		 * diff(x) = softmax(x) - label
		 */
		crossEntropyKernel.backward(x, label, diff);
		
		return diff;
	}
	
	public static void main(String[] args) {
		float[] x = MatrixUtils.order(20, 0.01f, 0.1f);
		Tensor xt = new Tensor(2, 1, 1, 10, x, true);
		float[] label = new float[] {0,1,0,0,0,0,0,0,0,0,
				0,0,0,0,0,0,0,0,1,0};
		Tensor labelt = new Tensor(2, 1, 1, 10, label, true);
		
		float max = MatrixOperation.max(x);
		
		float[] tmp = MatrixOperation.subtraction(x, max);
		
		float ln = (float) Math.log(MatrixOperation.sum(MatrixOperation.exp(tmp)));
		
		PrintUtils.printImage(MatrixOperation.subtraction(tmp, ln));
		
		Tensor loss = CrossEntropyLoss2.operation().loss(xt, labelt);
		
		PrintUtils.printImage(loss.syncHost());
		
		System.out.println();
		
		System.out.println("loss:"+JsonUtils.toJson(MatrixOperation.sum(loss.syncHost())/2));
		
		Tensor diff = CrossEntropyLoss2.operation().diff(xt, labelt);
		
		System.out.println("diff:"+JsonUtils.toJson(diff.syncHost()));
		
//		System.out.println(Math.log(Math.exp(-1.3470f)/sum));
//		
//		float d_yhat_k_x = yhat_k * (1 - yhat_k);
//		
//		float d_l_yhat_k = - 1 / yhat_k;
//		
//		System.out.println(d_yhat_k_x * d_l_yhat_k);
		
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

	@Override
	public Tensor loss(Tensor x, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		init(x);
		
		/**
		 * q(x) = softmax(x)
		 * H(p,q) = - ∑p(x)logq(x)
		 * 简化log_softmax:
		 * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
		 * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
		 */
		crossEntropyKernel.forward(x, label, loss);
		
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub

		/**
		 * diff(x) = softmax(x) - label
		 */
		crossEntropyKernel.backward(x, label, diff);
		
		return diff;
	}

	@Override
	public Tensor loss(Tensor x, Tensor label, int igonre) {
		// TODO Auto-generated method stub
		init(x);
		
		/**
		 * q(x) = softmax(x)
		 * H(p,q) = - ∑p(x)logq(x)
		 * 简化log_softmax:
		 * log(exp(xi)/sum(exp(X))) = (xi - max) - log(sum(exp(xi - max)))
		 * 该操作为了防止上溢出与下溢出情况导致nan与inf出现.
		 */
		crossEntropyKernel.forward(x, label, loss, igonre);
		
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, int igonre) {
		// TODO Auto-generated method stub
		/**
		 * diff(x) = softmax(x) - label
		 */
		crossEntropyKernel.backward(x, label, diff, igonre);
		
		return diff;
	}

}
