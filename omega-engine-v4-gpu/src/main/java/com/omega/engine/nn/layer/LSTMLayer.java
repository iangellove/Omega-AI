package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;

public class LSTMLayer extends Layer{
	
	private int time = 0;
	
	private int inputSize;
	
	private int hiddenSize;
	
	private boolean bias = false;
	
	private FullyLayer fxl;
	
	private FullyLayer ixl;
	
	private FullyLayer gxl;
	
	private FullyLayer oxl;
	
	private FullyLayer fhl;
	
	private FullyLayer ihl;
	
	private FullyLayer ghl;
	
	private FullyLayer ohl;
	
	private ActiveFunctionLayer fa;
	
	private ActiveFunctionLayer ia;
	
	private ActiveFunctionLayer ga;
	
	private ActiveFunctionLayer oa;
	
	private ActiveFunctionLayer ha;

	/**
	 * forgot gate
	 * ft = sigmoid(Wf * ht-1 + bhf + Uf * xt + bxf)
	 */
	private Tensor f;
	
	/**
	 * input gate
	 * it = sigmoid(Wi * ht-1 + bhi + Ui * xt + bxi)
	 */
	private Tensor i;
	
	/**
	 * candidate memory
	 * gt = tanh(Wg * ht-1 + bhg + Ug * xt + bxg)
	 */
	private Tensor g;

	/**
	 * cell status
	 * ct = ct-1 ⊙ ft + it ⊙ gt
	 */
	private Tensor c;
	
	/**
	 * output gate
	 * ot = sigmoid(Wo * ht-1 + bho + Uo * xt + bxo)
	 */
	private Tensor o;

	/**
	 * hidden status
	 * ht = ot ⊙ tanh(ct)
	 */
	private Tensor h;
	
	private Tensor temp;
	
	private BaseKernel baseKernel;
	
	public LSTMLayer(int inputNum,int hiddenNum,int time,boolean bias) {
		this.time = time;
		this.inputSize = inputNum;
		this.hiddenSize = hiddenNum;
		this.bias = bias;
		this.initLayers();
	}
	
	public LSTMLayer(int inputNum,int hiddenNum,int time,boolean bias,Network network) {
		this.network = network;
		this.time = time;
		this.inputSize = inputNum;
		this.hiddenSize = hiddenNum;
		this.bias = bias;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.fxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
		this.ixl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
		this.gxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
		this.oxl = FullyLayer.createRNNCell(inputSize, hiddenSize, time, bias, network);
		
		this.fhl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, bias, network);
		this.ihl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, bias, network);
		this.ghl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, bias, network);
		this.ohl = FullyLayer.createRNNCell(hiddenSize, hiddenSize, time, bias, network);
		
		this.fa = createActiveLayer(ActiveType.sigmoid, fhl);
		this.ia = createActiveLayer(ActiveType.sigmoid, ihl);
		this.ga = createActiveLayer(ActiveType.tanh, ghl);
		this.oa = createActiveLayer(ActiveType.sigmoid, ohl);
		this.ha = createActiveLayer(ActiveType.tanh, fhl);
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}

	}
	
	public ActiveFunctionLayer createActiveLayer(ActiveType activeType,Layer preLayer) {
		switch (activeType) {
		case sigmoid:
			return new SigmodLayer(preLayer);
		case relu:
			return new ReluLayer(preLayer);
		case leaky_relu:
			return new LeakyReluLayer(preLayer);
		case tanh:
			return new TanhLayer(preLayer);
		default:
			throw new RuntimeException("The rnn layer is not support the ["+activeType+"] active function.");
		}
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		RNN network = (RNN) this.network;
		this.time = network.time;
		if(this.h == null || this.h.number != this.number) {
			this.f = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.i = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.g = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.c = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.o = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.temp = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
			this.h = Tensor.createTensor(this.h, number, 1, 1, hiddenSize, true);
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		int batch = this.number / this.time;
		int onceSize = batch * this.h.getOnceSize();
		/**
		 * ft = f(W * xt + bx + U * fht-1 + bh)
		 * 
		 */
		if(this.input != null) {
			
			for(int t = 0;t<time;t++) {
				
				fxl.forward(this.input, batch, t);
				ixl.forward(this.input, batch, t);
				gxl.forward(this.input, batch, t);
				oxl.forward(this.input, batch, t);
				
				fhl.forward(this.f, batch, t - 1, t);
				ihl.forward(this.i, batch, t - 1, t);
				ghl.forward(this.g, batch, t - 1, t);
				ohl.forward(this.o, batch, t - 1, t);
				
				/**
				 * ft = sigmoid(Wf * ht-1 + bhf + Uf * xt + bxf)
				 * it = sigmoid(Wi * ht-1 + bhi + Ui * xt + bxi)
				 * gt = tanh(Wg * ht-1 + bhg + Ug * xt + bxg)
				 * ot = sigmoid(Wo * ht-1 + bho + Uo * xt + bxo)
				 */
				TensorOP.add(fxl.getOutput(), fhl.getOutput(), this.f, t * onceSize, onceSize);
				TensorOP.add(ixl.getOutput(), ihl.getOutput(), this.i, t * onceSize, onceSize);
				TensorOP.add(gxl.getOutput(), ghl.getOutput(), this.g, t * onceSize, onceSize);
				TensorOP.add(oxl.getOutput(), ohl.getOutput(), this.o, t * onceSize, onceSize);
				
				fa.forward(this.f, batch, t);
				ia.forward(this.i, batch, t);
				ga.forward(this.g, batch, t);
				oa.forward(this.o, batch, t);

				/**
				 * ct-1 ⊙ ft + it ⊙ gt
				 */
				TensorOP.mul(i, g, temp, t * onceSize, onceSize);
				TensorOP.mul(f, c, c, t * onceSize, onceSize);
				TensorOP.add(temp, c, c, t * onceSize, onceSize);
				
				/**
				 * ht = ot ⊙ tanh(ct)
				 */
				ha.forward(c, batch, t);
				TensorOP.mul(o, ha.getOutput(), this.h, t * onceSize, onceSize);
				
//				baseKernel.copy_gpu(ha.getOutput(), this.h, onceSize, t * onceSize, 1, t * onceSize, 1);
	
			}
		}

		this.output = this.h;
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		int batch = this.number / time;
		int onceSize = batch * fhl.input.getOnceSize();

		fxl.clear();
		ixl.clear();
		gxl.clear();
		oxl.clear();
		
		fhl.clear();
		ihl.clear();
		ghl.clear();
		ohl.clear();
		
//		for(int t = time-1;t>=0;t--) {
//
//			if(t < time - 1) {
//				baseKernel.axpy_gpu(selfLayer.diff, this.delta, onceSize, 1, t * onceSize, 1, t * onceSize, 1);
//			}
//			
//			outputActive.back(this.delta, batch, t);
//			
//			selfLayer.back(outputActive.diff, batch, t, t, t - 1);
//
//			inputLayer.back(outputActive.diff, batch, t);
//			
//		}
//		
//		this.diff = inputLayer.diff;
//		this.diff.showDM(0);
//		inputLayer.diffW.showDM(0);
//		selfLayer.diffW.showDM(0);
//		inputLayer.diffB.showDM(0);
//		selfLayer.diffB.showDM(0);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 计算输出
		 */
		this.output();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(inpnut);
		/**
		 * 计算输出
		 */
		this.output();
		
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
//		inputLayer.update(number / time);
//		selfLayer.update(number / time);
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.rnn;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}
}
