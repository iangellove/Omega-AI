package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.layer.gpu.RNNBaseKernel;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;

/**
 * Recurrent Layer
 * @author Administrator
 *
 */
public class RNNLayer extends Layer{
	
	private int time = 0;
	
	private int inputSize;
	
	private int hiddenSize;
	
	private int outputSize;
	
	private boolean batchNormal = false;
	
	private ActiveType activeType;
	
	private FullyLayer inputLayer;
	
	private FullyLayer selfLayer;
	
	private FullyLayer outputLayer;
	
	private BNLayer inputBN;
	
	private BNLayer selfBN;
	
	private BNLayer outputBN;
	
	private ActiveFunctionLayer inputActive;
	
	private ActiveFunctionLayer selfActive;
	
	private ActiveFunctionLayer outputActive;
	
	private Tensor h;
	
	private BaseKernel baseKernel;
	
//	private RNNBaseKernel rnnKenrel;
	
	public RNNLayer(int inputNum,int hiddenNum,int outputNum,int time,ActiveType activeType,boolean batchNormal) {
		this.time = time;
		this.inputSize = inputNum;
		this.hiddenSize = hiddenNum;
		this.outputSize = outputNum;
		this.activeType = activeType;
		this.batchNormal = batchNormal;
		this.initLayers();
	}
	
	public RNNLayer(int inputNum,int hiddenNum,int outputNum,int time,ActiveType activeType,boolean batchNormal,Network network) {
		this.network = network;
		this.time = time;
		this.inputSize = inputNum;
		this.hiddenSize = hiddenNum;
		this.outputSize = outputNum;
		this.activeType = activeType;
		this.batchNormal = batchNormal;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.inputLayer = new FullyLayer(inputSize, hiddenSize, !batchNormal, this.network);

		this.selfLayer = new FullyLayer(hiddenSize, hiddenSize, !batchNormal, this.network);
		
		this.outputLayer = new FullyLayer(hiddenSize, outputSize, !batchNormal, this.network);

		if (batchNormal) {
			this.inputBN = new BNLayer(inputLayer);
			this.selfBN = new BNLayer(selfLayer);
			this.outputBN = new BNLayer(outputLayer);
			this.inputActive = createActiveLayer(activeType, inputBN);
			this.selfActive = createActiveLayer(activeType, selfBN);
			this.outputActive = createActiveLayer(activeType, outputBN);
		}else {
			this.inputActive = createActiveLayer(activeType, inputLayer);
			this.selfActive = createActiveLayer(activeType, selfLayer);
			this.outputActive = createActiveLayer(activeType, outputLayer);
		}
		
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
		case silu:
			return new SiLULayer(preLayer);
		default:
			throw new RuntimeException("The cbl layer is not support the ["+activeType+"] active function.");
		}
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		int hSize = this.number / time * (time + 1);
		if(this.h == null || this.h.number != hSize) {
			this.h = new Tensor(hSize, 1, 1, hiddenSize, true);
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
		int batch = this.number / time;
		int onceSize = batch * this.h.getOnceSize();
		/**
		 * ht = f(W * ht-1 + U * xt + bh)
		 * yt = f(V * ht + by)
		 */
		this.h.clearGPU();
		if(this.input != null) {
			
			for(int t = 0;t<time;t++) {
				
				inputLayer.forward(this.input, batch, t);
				Tensor o1 = inputLayer.output;
				if(batchNormal) {
					inputBN.forward(o1, batch, t);
					o1 = inputBN.output;
				}
				inputActive.forward(o1, batch, t);
				
				selfLayer.forward(this.h, batch, t);
				Tensor o2 = selfLayer.output;
				if(batchNormal) {
					selfBN.forward(o2, batch, t);
					o2 = selfBN.output;
				}
				selfActive.forward(o2, batch, t);
				
				baseKernel.axpy_gpu(inputActive.getOutput(), this.h, onceSize, 1, t * onceSize, 1, (t + 1) * onceSize, 1);
				
				baseKernel.axpy_gpu(selfActive.getOutput(), this.h, onceSize, 1, t * onceSize, 1, (t + 1) * onceSize, 1);

				outputLayer.forward(this.h, batch, t + 1, t);
				Tensor o3 = outputLayer.output;
				if(batchNormal) {
					outputBN.forward(o3, batch, t);
					o3 = outputBN.output;
				}
				outputActive.forward(o3, batch, t);
//				outputActive.getOutput().showDM();
			}
		}
		this.output = outputActive.output;
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
		int onceSize = batch * inputLayer.input.getOnceSize();
		/**
		 * E = ∑et
		 * du = ∑de/du
		 * dv = ∑de/dv
		 * dw = ∑de/dw
		 * 
		 * dv = htT * deltadarknet源码解读
		 * dbv = delta
		 * dhtt = vT * delta + wT * A't+1 * dht+1
		 * if t = t end
		 * dhtt = vT * delta
		 * dw = A't * dhtt * ht-1T
		 * dbu = A't * dhtt
		 * du = A't * dhtt * xtT
		 */
		outputLayer.clear();
		selfLayer.clear();
		inputLayer.clear();
		if(inputLayer.diff != null) {
			inputLayer.diff.clearGPU();
		}
		if(selfLayer.diff != null) {
			selfLayer.diff.clearGPU();
		}
		if(outputLayer.diff != null) {
			outputLayer.diff.clearGPU();
		}
		for(int t = time-1;t>=0;t--) {
			
			outputActive.back(this.delta, batch, t);
			Tensor d3 = outputActive.diff;
			if (batchNormal) {
				outputBN.back(d3, batch, t);
				d3 = outputBN.diff;
			}
			outputLayer.back(d3, batch, t + 1, t);

			if(selfLayer.diff != null) {
				baseKernel.axpy_gpu(selfLayer.diff, outputLayer.diff, onceSize, 1, t * onceSize, 1, t * onceSize, 1);
			}
			
			if(t > 0) {
				selfActive.back(outputLayer.diff, batch, t);
				Tensor d2 = selfActive.diff;
				if (batchNormal) {
					selfBN.back(d2, batch, t);
					d2 = selfBN.diff;
				}
				selfLayer.back(d2, batch, t, t, t - 1);
//				selfLayer.diff.showDM();
			}

			inputActive.back(outputLayer.diff, batch, t);
			Tensor d1 = inputActive.diff;
			if (batchNormal) {
				inputBN.back(d1, batch, t);
				d1 = inputBN.diff;
			}
			inputLayer.back(d1, batch, t);
		}
		
		this.diff = inputLayer.diff;
//		outputLayer.diffW.showDM();
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
		inputLayer.update();
		selfLayer.update();
		outputLayer.update();
		if(batchNormal) {
			inputBN.update();
			selfBN.update();
			outputBN.update();
		}
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
