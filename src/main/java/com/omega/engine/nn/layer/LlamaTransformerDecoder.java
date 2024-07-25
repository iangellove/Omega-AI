package com.omega.engine.nn.layer;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.utils.ModelUtils;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class LlamaTransformerDecoder extends Layer{
	
	private int time;
	
	private int vocab_size;
	
	private int embedDim = 0;
	
	private boolean flashAttention = false;
	
	private boolean bias = false;
	
	private boolean dropout = false;
	
	private int headNum = 8;
	
	private int n_layers = 6;
	
	private EmbeddingIDLayer src_emb;
	private List<LlamaTransformerBlock> decoderLayers;
	private RMSLayer norm;
	private DropoutLayer dropoutLayer;
	
	private BaseKernel baseKernel;
	
	public LlamaTransformerDecoder(int vocab_size,int n_layers,int headNum,int time,int embedDim,boolean bias,boolean dropout) {
		this.headNum = headNum;
		this.n_layers = n_layers;
		this.vocab_size = vocab_size;
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.channel = 1;
		this.height = 1;
		this.width = embedDim;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public LlamaTransformerDecoder(int vocab_size,int n_layers,int headNum,int time,int embedDim,boolean bias,boolean dropout,boolean flashAttention,Network network) {
		this.flashAttention = flashAttention;
		this.headNum = headNum;
		this.n_layers = n_layers;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.vocab_size = vocab_size;
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.channel = 1;
		this.height = 1;
		this.width = embedDim;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.src_emb = new EmbeddingIDLayer(vocab_size, embedDim, network);
		this.src_emb.weight = new Tensor(1, 1, src_emb.width, src_emb.oWidth, RandomUtils.uniform(this.src_emb.width * this.src_emb.oWidth, 0.0f, 0.02f), true);
//		this.src_emb.weight = new Tensor(1, 1, src_emb.width, src_emb.oWidth, RandomUtils.order(this.src_emb.width * this.src_emb.oWidth, 0.001f, 0.001f), true);

		decoderLayers = new ArrayList<LlamaTransformerBlock>();
		
		for(int i = 0;i<n_layers;i++) {
			LlamaTransformerBlock decoderLayer = new LlamaTransformerBlock(headNum, time, embedDim, bias, dropout, flashAttention, network);
			decoderLayers.add(decoderLayer);
		}
		
		this.norm = new RMSLayer(decoderLayers.get(n_layers - 1));
		
		if(dropout) {
			dropoutLayer = new DropoutLayer(0.1f, src_emb);
		}
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		this.time = this.network.time;
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

	}
	
	public void output(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
	
		src_emb.forward(input);

		Tensor out1 = src_emb.getOutput();
		
		if(dropout) {
			this.dropoutLayer.forward(out1);
			out1 = dropoutLayer.getOutput();
		}
		
		for(int i = 0;i<n_layers;i++) {
			decoderLayers.get(i).forward(cos, sin, out1);
			out1 = decoderLayers.get(i).getOutput();
		}
		
		this.norm.forward(out1);
		this.output = this.norm.getOutput();
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
	
	public void diff(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		
		this.norm.back(delta);
		Tensor decoderDiff = this.norm.diff;
		
		for(int i = n_layers - 1;i>=0;i--) {
			decoderLayers.get(i).back(cos, sin, decoderDiff);
			decoderDiff = decoderLayers.get(i).diff;
		}
		
		if(dropout) {
			this.dropoutLayer.back(decoderDiff);
			decoderDiff = dropoutLayer.diff;
		}
//		System.err.println("decoderDiff:");
//		decoderDiff.showDM();
		src_emb.back(decoderDiff);

		this.diff = this.src_emb.diff;
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub

	}
	
	@Override
	public void back() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		
	}
	
	public void forward(Tensor cos,Tensor sin,Tensor input) {
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
		this.output(cos, sin);
		
	}
	
	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

	}
	
	public void back(Tensor cos,Tensor sin,Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(cos, sin);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		src_emb.update();
		norm.update();
		for(int i = 0;i<n_layers;i++) {
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
		src_emb.saveModel(outputStream);
		
		for(int i = 0;i<n_layers;i++) {
			decoderLayers.get(i).saveModel(outputStream);
		}
		
		norm.saveModel(outputStream);
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		
		src_emb.loadModel(inputStream);
		
		for(int i = 0;i<n_layers;i++) {
			decoderLayers.get(i).loadModel(inputStream);
		}
		
		norm.loadModel(inputStream);
		
	}
	
}
