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
public class CrossEntropyLossIdx extends LossFunction {

	public final LossType lossType = LossType.softmax_with_cross_entropy_idx;
	
	private static CrossEntropyLossIdx instance;
	
//	private Tensor output;
	
	private Tensor loss;
	
	private Tensor probs;
	
	private Tensor diff;
	
//	private SoftmaxKernel softmaxKernel;
	
	private CrossEntropyKernel crossEntropyKernel;
	
	public CrossEntropyLossIdx() {
		initKernel();
	}
	
	public static CrossEntropyLossIdx operation() {
		if(instance == null) {
			instance = new CrossEntropyLossIdx();
		}
		return instance;
	}
	
	public void init(Tensor input) {
		if(loss == null || loss.number != input.number) {
			this.loss = new Tensor(input.number, 1, 1, 1, true);
			this.probs = new Tensor(input.number, input.channel, input.height, input.width, true);
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
//		crossEntropyKernel.softmax(x, probs);
//		
//		crossEntropyKernel.crossentropy(probs, label, loss);
		
		crossEntropyKernel.forwardIDX2(x, label, probs, loss, -99999);
		
		return loss;
	}
	
	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		
		/**
		 * diff(x) = softmax(x) - label
		 */
//		crossEntropyKernel.crossentropy_backward(probs, label, diff);
		
		crossEntropyKernel.backwardIDX2(probs, label, diff, -99999);
		
		return diff;
	}
	
	public static void main(String[] args) {
		float[] x = MatrixUtils.order(20, 0.01f, 0.1f);
		Tensor xt = new Tensor(2, 1, 1, 10, x, true);
		float[] label = new float[] {1, 8};
		Tensor labelt = new Tensor(2, 1, 1, 1, label, true);
		
		float max = MatrixOperation.max(x);
		
		float[] tmp = MatrixOperation.subtraction(x, max);
		
		float ln = (float) Math.log(MatrixOperation.sum(MatrixOperation.exp(tmp)));
		
		PrintUtils.printImage(MatrixOperation.subtraction(tmp, ln));
		
		Tensor loss = CrossEntropyLossIdx.operation().loss(xt, labelt);
		
		PrintUtils.printImage(loss.syncHost());
		
		System.out.println();
		
		System.out.println("loss:"+JsonUtils.toJson(MatrixOperation.sum(loss.syncHost())/2));
		
		Tensor diff = CrossEntropyLossIdx.operation().diff(xt, labelt);
		
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
//		crossEntropyKernel.softmax(x, probs);
//		
//		crossEntropyKernel.crossentropy(probs, label, loss);
		
		crossEntropyKernel.forwardIDX2(x, label, probs, loss, -99999);
		
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub

		/**
		 * diff(x) = softmax(x) - label
		 */
//		crossEntropyKernel.crossentropy_backward(probs, label, diff);
		
		crossEntropyKernel.backwardIDX2(probs, label, diff, -99999);
		
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
//		crossEntropyKernel.softmax(x, probs);
//		probs.showDM(0);
//		JCuda.cudaDeviceSynchronize();
////		probs.showDMByNumber(0);
//		crossEntropyKernel.crossentropy_igone(probs, label, loss, igonre);
//		JCuda.cudaDeviceSynchronize();
//		loss.showDM();
		
		crossEntropyKernel.forwardIDX2(x, label, probs, loss, igonre);
		
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label, int igonre) {
		// TODO Auto-generated method stub
		/**
		 * diff(x) = softmax(x) - label
		 */
//		probs.showDMByNumber(0);
//		crossEntropyKernel.crossentropy_backward_igone(probs, label, diff, igonre);
		
		crossEntropyKernel.backwardIDX2(probs, label, diff, igonre);
		
		return diff;
	}

}
