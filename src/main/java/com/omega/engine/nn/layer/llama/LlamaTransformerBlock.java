package com.omega.engine.nn.layer.llama;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.RMSLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class LlamaTransformerBlock extends Layer{
	
	private int time;
	
	private int headNum = 8;
	
	private int embedDim = 0;
	
	private boolean bias = false;
	
	private boolean dropout = false;
	
	private boolean flashAttention = false;
	
	private LlamaAttentionLayer attn;
	
	private RMSLayer norm1;
	
	/**
	 * mlp
	 */
	private LlamaMLPLayer mlp;
	private RMSLayer norm2;
	
	private BaseKernel baseKernel;
	
	private Tensor tmp1;
	
	private Tensor tmp2;
	
	public LlamaTransformerBlock(int headNum,int time,int embedDim,boolean bias,boolean dropout) {
		this.headNum = headNum;
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public LlamaTransformerBlock(int headNum,int time,int embedDim,boolean bias,boolean dropout,boolean flashAttention,Network network) {
		this.flashAttention = flashAttention;
		this.headNum = headNum;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.dropout = dropout;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {

		this.norm1 = new RMSLayer(this);
		
		if(flashAttention) {
			this.attn = new LlamaFlashAttentionLayer(embedDim, headNum, time, bias, dropout, network);
		}else {
			this.attn = new LlamaCausalSelfAttentionLayer(embedDim, headNum, time, bias, dropout, network);
		}

		this.norm2 = new RMSLayer(attn);
		
		this.mlp = new LlamaMLPLayer(embedDim, embedDim, bias, network);
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}

	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		this.time = this.network.time;
		if(this.tmp1 == null || this.tmp1.number != this.number) {
//			if(this.tmp1 == null) {
//				System.out.println(number+":"+embedDim);
//			}
			this.tmp1 = Tensor.createGPUTensor(this.tmp1, number, 1, 1, embedDim, true);
			this.tmp2 = Tensor.createGPUTensor(this.tmp2, number, 1, 1, embedDim, true);
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

	}
	
	public void output(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		
		norm1.forward(input);

		attn.forward(cos, sin, norm1.getOutput());

		TensorOP.add(attn.getOutput(), input, tmp1);

		norm2.forward(tmp1);
		
		mlp.forward(norm2.getOutput());
		
		TensorOP.add(mlp.getOutput(), tmp1, tmp2);
		
		this.output = tmp2;
//		System.err.println("---------------------------------");
//		this.output.showDM();
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
		
//		delta.showDM();
		
		mlp.back(delta);

		norm2.back(mlp.diff);
		
		TensorOP.add(norm2.diff, delta, norm2.diff);
		
//		norm2.diff.showDM();
		
		attn.back(cos, sin, norm2.diff);
		
		norm1.back(attn.diff);
		
		TensorOP.add(norm1.diff, norm2.diff, tmp2);
		
		this.diff = tmp2;
//		System.err.println("diff:");
//		diff.showDM();
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
		norm1.update();
		attn.update();
		norm2.update();
		mlp.update();
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
		norm1.saveModel(outputStream);
		attn.saveModel(outputStream);
		norm2.saveModel(outputStream);
		mlp.saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		norm1.loadModel(inputStream);
		attn.loadModel(inputStream);
		norm2.loadModel(inputStream);
		mlp.loadModel(inputStream);
	}
	
}
