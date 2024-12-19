package com.omega.engine.nn.layer.clip.bert;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * BertAttention Layer
 * @author Administrator
 *
 */
public class BertAttentionLayer extends Layer{
	
	private int time;
	
	private int headNum;
	
	private int hiddenSize;
	
	public BertSelfAttentionLayer attn;
	public BertAttnOutputLayer out;

	public BertAttentionLayer(int headNum,int time,int hiddenSize) {
		this.headNum = headNum;
		this.time = time;
		this.hiddenSize = hiddenSize;
		this.channel = 1;
		this.height = 1;
		this.width = hiddenSize;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = hiddenSize;
		this.initLayers();
	}
	
	public BertAttentionLayer(int headNum,int time,int hiddenSize,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.headNum = headNum;
		this.time = time;
		this.hiddenSize = hiddenSize;
		this.channel = 1;
		this.height = 1;
		this.width = hiddenSize;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = hiddenSize;
		this.initLayers();
	}
	
	public void initLayers() {

		this.attn = new BertSelfAttentionLayer(hiddenSize, headNum, time, true, false, network);

		this.out = new BertAttnOutputLayer(hiddenSize, hiddenSize, network);
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
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

		attn.forward(input);

		out.forward(attn.getOutput(), input);
		
		this.output = out.getOutput();

	}
	
	public void output(Tensor mask) {
		// TODO Auto-generated method stub

		attn.forward(input, mask);
		
		out.forward(attn.getOutput(), input);

		this.output = out.getOutput();

	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 参数初始化
		 */
		this.init();
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
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 计算输出
		 */
		this.output();
		
	}
	
	public void forward(Tensor input,Tensor mask) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 计算输出
		 */
		this.output(mask);
		
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

	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.bert_output;
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {

	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {

	}
	
	public static void main(String[] args) {
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub

	}

}