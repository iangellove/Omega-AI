package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.gpu.DropoutKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;

/**
 * Dropout Layer
 * @author Administrator
 *
 */
public class DropoutLayer extends Layer {
	
	private float probability = 0.5f;
	
	private Tensor mask;
	
	public Layer preLayer;
	
	private float scale = 0.0f;
	
	private DropoutKernel kernel;
	
	public DropoutLayer(float probability) {
		this.probability = probability;
		this.scale = 1.0f / (1.0f - probability);
	}
	
	public DropoutLayer(float probability,Layer preLayer) {
		this.setPreLayer(preLayer);
		this.probability = probability;
		this.scale = 1.0f / (1.0f - probability);
	}
	
	public DropoutLayer(float probability,Network network) {
		this.network = network;
		this.probability = probability;
		this.scale = 1.0f / (1.0f - probability);
	}
	
	public void setPreLayer(Layer pre) {
		this.preLayer = pre;
		this.network = pre.network;
		this.channel = preLayer.oChannel;
		this.height = preLayer.oHeight;
		this.width = preLayer.oWidth;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		
		if(kernel == null) {
			kernel = new DropoutKernel(this.probability, this.scale);
		}
		this.number = this.network.number;
		initParam();
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

		/**
		 * 训练
		 */
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			if(this.mask == null || this.mask.number != this.number) {
				this.mask = Tensor.createTensor(this.mask, number, channel, height, oWidth, true);
			}
			GPUOP.getInstance().cudaRandom(this.mask);
		}

	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			kernel.forward(input, mask);
		}
		this.output = input;
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			kernel.backward(delta, mask);
		}
		this.diff = delta;
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
		initBack();
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
	public void update() {
		// TODO Auto-generated method stub

	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.dropout;
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
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(input);
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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

}
