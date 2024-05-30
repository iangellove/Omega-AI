package com.omega.engine.ad.op.functions;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.FunctionOP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.GPUOP;

public class TransposeOP extends FunctionOP {

	/**
	 * 
	 */
	private static final long serialVersionUID = -3857343378511617891L;

	public static TransposeOP op = null;
	
	public static final OPType opt = OPType.transpose;
	
	public static TransposeOP getInstance() {
		if(op == null) {
			op = new TransposeOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	@Override
	public Tensor forward(Tape tape) {
		// TODO Auto-generated method stub
		Tensor self = tape.getX();
		Tensor y = tape.getOutput();
		TensorOP.transpose(self, y);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		return y;
	}
	
	/**
	 * xt' = deltat
	 */
	@Override
	public void backward(Tensor delta, Tape tape) {
		// TODO Auto-generated method stub
		Tensor x = tape.getX();
		if(x.isRequiresGrad()) {
			Tensor dy = tape.getTmp();
			TensorOP.transpose(delta, dy);
			TensorOP.mulPlus(dy, 1.0f, x.getGrad());
		}
	}
	
	public static void main(String[] args) {
		
		testPermute();
		
	}
	
	public static void testPermute() {
	    	
	    	int batch = 2;
	    	int m = 5;
	    	int n = 2;
	    	int k = 2;
	    	
	    	float[] a = RandomUtils.order(batch * m * n * k, 1, 0);
	    	
	    	Tensor at = new Tensor(batch, m, n, k, a, true);
	    	
	    	Tensor ct = new Tensor(batch, n, m, k, true);
	    	
	    	at.showDM();
	    	
	    	TensorOP.permute(at, ct, new int[] {0, 2, 1, 3});
	    	
	    	ct.showDM();
	    	
	}
	
}
