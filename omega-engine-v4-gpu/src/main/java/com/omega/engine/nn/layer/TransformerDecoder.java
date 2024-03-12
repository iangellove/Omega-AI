package com.omega.engine.nn.layer;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.network.Network;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class TransformerDecoder extends Layer{
	
	private int time;
	
	private int vocab_size;
	
	private int embedDim = 0;
	
	private int nChannel = 1;
	
	private boolean bias = false;
	
	private boolean layer_norm = false;
	
	private int headNum = 8;
	
	private int n_layers = 1;
	
	private EmbeddingLayer src_emb;
	private EmbeddingLayer pos_emb;
	private List<TransformerDecoderLayer> decoderLayers;
	
	private BaseKernel baseKernel;
	
	private Tensor positions;


	public TransformerDecoder(int vocab_size,int time,int embedDim,int nChannel,boolean bias,boolean layer_norm) {
		this.vocab_size = vocab_size;
		this.time = time;
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.initLayers();
	}
	
	public TransformerDecoder(int vocab_size,int time,int embedDim,int nChannel,boolean bias,boolean layer_norm,Network network) {
		this.network = network;
		this.vocab_size = vocab_size;
		this.time = time;
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.layer_norm = layer_norm;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.src_emb = new EmbeddingLayer(vocab_size, embedDim, network);
		this.pos_emb = new EmbeddingLayer(time, embedDim, network);
		
		decoderLayers = new ArrayList<TransformerDecoderLayer>();
		
		for(int i = 0;i<n_layers;i++) {
			TransformerDecoderLayer decoderLayer = new TransformerDecoderLayer(time, embedDim, nChannel, bias, layer_norm, network);
			decoderLayers.add(decoderLayer);
		}
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
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

		src_emb.forward(input);
		
		pos_emb.forward(positions);
		
		TensorOP.add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
		
		for(int i = 0;i<n_layers;i++) {
			decoderLayers.get(i).forward(src_emb.getOutput());
		}
		
		this.output = decoderLayers.get(n_layers - 1).getOutput();
		
	}
	
	public void output(Tensor mask,Tensor positions) {
		// TODO Auto-generated method stub

		src_emb.forward(input);
		
		pos_emb.forward(positions);
		
		TensorOP.add(src_emb.getOutput(), pos_emb.getOutput(), src_emb.getOutput());
		
		Tensor decoderOutput = src_emb.getOutput();
		
		for(int i = 0;i<n_layers;i++) {
			decoderLayers.get(i).forward(decoderOutput, mask);
			decoderOutput = decoderLayers.get(i).getOutput();
		}
		
		this.output = decoderOutput;
		
	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		Tensor decoderDiff = delta;
		for(int i = n_layers - 1;i>=0;i--) {
			decoderLayers.get(i).back(decoderDiff);
			decoderDiff = decoderLayers.get(i).diff;
		}
		
		pos_emb.back(decoderDiff);
		
		src_emb.back(decoderDiff);

		this.diff = this.src_emb.diff;
		
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
	
	public void forward(Tensor input,Tensor mask,Tensor positions) {
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
		this.output(mask, positions);
		
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
		src_emb.update();
		pos_emb.update();
		for(int i = 0;i<n_layers - 1;i++) {
			decoderLayers.get(i).update();
		}
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
	
}
