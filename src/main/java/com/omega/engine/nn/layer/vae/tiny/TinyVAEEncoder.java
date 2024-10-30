package com.omega.engine.nn.layer.vae.tiny;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class TinyVAEEncoder extends Layer {

	public TinyVAEConvBlock block1;
	public TinyVAEConvBlock block2;
	public TinyVAEConvBlock block3;
	
	public TinyVAEEncoder(int channel,int height,int width, Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = 64;
		this.height = height;
		this.width = width;
		
		initLayers();
		
		this.oHeight = block3.oHeight;
		this.oWidth = block3.oWidth;
	}
	
	public void initLayers() {
		block1 = new TinyVAEConvBlock(channel, 64, height, width, network);
		block2 = new TinyVAEConvBlock(64, 128, block1.oHeight, block1.oWidth, network);
		block3 = new TinyVAEConvBlock(128, 256, block2.oHeight, block2.oWidth, network);
	}

	@Override
	public void init() {
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		block1.forward(this.input);
		block2.forward(block1.getOutput());
		block3.forward(block2.getOutput());
		
		this.output = block3.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		block3.back(this.delta);
		block2.back(block3.diff);
		block1.back(block2.diff);
		
		this.diff = block1.diff;
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

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		block1.update();
		block2.update();
		block3.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.block;
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

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();

	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

}
