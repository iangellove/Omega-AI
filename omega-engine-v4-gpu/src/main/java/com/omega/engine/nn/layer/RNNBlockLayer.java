package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.cudnn.RNNCudnnKernel;
import com.omega.engine.nn.layer.gpu.RNNBaseKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;

import jcuda.Sizeof;

/**
 * Recurrent Layer
 * @author Administrator
 *
 */
public class RNNBlockLayer extends Layer{
	
	private int time = 0;
	
	private int inputSize;
	
	private int hiddenSize;
	
	private int layerNum = 1;

	private RNNBaseKernel kernel;
	
	private int rnnMode = 0;
	
	private boolean bidirectional = false;
	
	private float dropout = 0;
	
//	private int numLinearLayers = 2;
	
	public RNNBlockLayer(int time,int inputSize,int hiddenSize,int rnnMode,boolean bidirectional,float dropout) {
		this.hasBias = false;
		this.time = time;
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.rnnMode = rnnMode;
		this.bidirectional = bidirectional;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = hiddenSize;
		this.initKernel();
	}
	
	public RNNBlockLayer(int time,int layerNum,int inputSize,int hiddenSize,int rnnMode,boolean bidirectional,float dropout,Network network) {
		this.hasBias = false;
		this.layerNum = layerNum;
		this.network = network;
		this.time = time;
		this.inputSize = inputSize;
		this.hiddenSize = hiddenSize;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = hiddenSize;
		this.rnnMode = rnnMode;
		this.bidirectional = bidirectional;
		this.dropout = dropout;
		this.initKernel();
	}
	
	public void initKernel() {
		
		if(kernel == null) {
			kernel = new RNNCudnnKernel(time, layerNum, inputSize, hiddenSize, bidirectional, rnnMode, dropout);
		}

	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		RNN network = (RNN) this.network;
		this.time = network.time;
		if(this.time != kernel.seqLength) {
			kernel.seqLength = this.time;
		}
		if(this.output == null || this.number != this.output.number){
			this.output = new Tensor(number, 1, 1, hiddenSize, true);
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, 1, 1, inputSize, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
		if(this.weight == null) {

			int weightSize = (int) (kernel.weightSize() / Sizeof.FLOAT);

//			this.weight = new Tensor(1, 1, 1, weightSize, RandomUtils.kaiming_uniform(weightSize, inputSize, this.paramsInit), true);
			
			this.weight = new Tensor(1, 1, 1, weightSize, true);
			
			this.diffW = new Tensor(1, 1, 1, weightSize, true);
			
			kernel.initWeights(weight);
			
//			weight.showDM();
			
		}
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		kernel.init(input.number);
		
		initParam();

		kernel.forward(input, weight, output);
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		System.out.println("in-rnn");
		kernel.dx(delta, output, weight, diff);
//		GradClipping.gradClipping(diff, 1e-7f);
		kernel.dw(delta, output, input, diffW);
		
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
		if(!this.freeze) {
			if(this.updater != null){
				this.updater.update(this);
			}else{
				
				for(int i = 0;i<this.weight.getDataLength();i++) {
					this.weight.data[i] -= this.learnRate * this.diffW.data[i];
				}
				
				for(int i = 0;i<this.bias.getDataLength();i++) {
					this.bias.data[i] -= this.learnRate * this.diffB.data[i];
				}
				
			}
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
