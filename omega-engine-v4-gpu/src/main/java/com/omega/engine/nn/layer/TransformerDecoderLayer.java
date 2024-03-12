package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.nn.network.Network;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class TransformerDecoderLayer extends Layer{
	
	private int time;
	
	private int embedDim = 0;
	
	private int nChannel = 1;
	
	private boolean bias = false;
	
	private boolean layer_norm = false;
	
	private int headNum = 8;
	
	private MultiHeadAttentionLayer attn;
	private PoswiseFeedForwardLinearLayer feed_forward;
//	private LNLayer norm1;
//	private LNLayer norm2;

	public TransformerDecoderLayer(int time,int embedDim,int nChannel,boolean bias,boolean layer_norm) {
		this.time = time;
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.initLayers();
	}
	
	public TransformerDecoderLayer(int time,int embedDim,int nChannel,boolean bias,boolean layer_norm,Network network) {
		this.network = network;
		this.time = time;
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.attn = new MultiHeadAttentionLayer(embedDim, headNum, time, bias, layer_norm, network);
		this.feed_forward = new PoswiseFeedForwardLinearLayer(embedDim, nChannel, bias, layer_norm, network);
//		this.norm1 = new LNLayer(this.attn);
//		this.norm2 = new LNLayer(this.feed_forward);
//		
//		if(baseKernel == null) {
//			baseKernel = new BaseKernel();
//		}
		
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
		
		this.attn.forward(input);
		
		this.feed_forward.forward(this.attn.getOutput());
		
		this.output = this.feed_forward.getOutput();
		
	}
	
	public void output(Tensor mask) {
		// TODO Auto-generated method stub
		
		this.attn.forward(input, mask);
		
		this.feed_forward.forward(this.attn.getOutput());
		
		this.output = this.feed_forward.getOutput();
		
	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		this.feed_forward.back(delta);
		
		this.attn.back(feed_forward.diff);

		this.diff = this.attn.diff;
		
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
		attn.update();
		feed_forward.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.transformer_decoder;
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
	
	public static void main(String[] args) {
		
		int batchSize = 5;
		int time = 10;
		int embedDim = 8;

		int nChannel = 4;
		
		CNN tf = new CNN(null);
		tf.CUDNN = true;
		tf.number = batchSize;
		
		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
		
		Tensor input = new Tensor(batchSize , time, 1, embedDim, data, true);
		
		input.showShape();
		input.showDM();
		
		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		Tensor delta = new Tensor(batchSize, time, 1, embedDim, delta_data, true);
		
		TransformerDecoderLayer mal = new TransformerDecoderLayer(time, embedDim, nChannel, false, true, tf);
		
//		mal.forward(input);
		
		mal.forward(input);
		
//		input.showDM();
//		mal.getWeights().showDM();
//		mal.getOutput().showShape();
		
		mal.getOutput().showDM();
		
		mal.back(delta);
//		mal.diff.showShape();
		mal.diff.showDM();
		
	}

}
