package com.omega.engine.nn.layer.clip;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * CLIPTextEmbeddingLayer
 * @author Administrator
 *
 */
public class CLIPTextEmbeddingLayer extends Layer{
	
	private int embedDim = 0;
	
	private int vocabSize;
	
	private int maxPositionEmbeddings;

	private EmbeddingIDLayer tokenEmbedding;
	
	private EmbeddingIDLayer positionEmbedding;
	
	private Tensor positionIDS;
	
	private BaseKernel kernel;
	
	public CLIPTextEmbeddingLayer(int vocabSize,int embedDim,int maxPositionEmbeddings,Network network) {
		this.network = network;
		this.vocabSize = vocabSize;
		this.maxPositionEmbeddings = maxPositionEmbeddings;
		this.embedDim = embedDim;
		
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		
		initLayers();
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = network.number;
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;

	}
	
	public void initLayers() {
		
		this.tokenEmbedding = new EmbeddingIDLayer(vocabSize, embedDim, false, network);

		this.positionEmbedding = new EmbeddingIDLayer(maxPositionEmbeddings, embedDim, false, network);

		if(positionIDS == null) {
			float[] data = RandomUtils.order(maxPositionEmbeddings, 1.0f, 0.0f);
			positionIDS = new Tensor(maxPositionEmbeddings, 1, 1, 1, data, true);
		}
		
		if(kernel == null) {
			kernel = new BaseKernel();
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

		this.getTokenEmbedding().forward(this.input);

		this.positionEmbedding.forward(positionIDS);
		
		TensorOP.addAxis(getTokenEmbedding().getOutput(), positionEmbedding.getOutput(), getTokenEmbedding().getOutput(), positionEmbedding.getOutput().getDataLength());
		
		this.output = getTokenEmbedding().getOutput();
		
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
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
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
		return LayerType.clip_vision_embedding;
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
		
		int batchSize = 2;
		int time = 512;
		int embedDim = 6;
		int vocabSize = 512;
		int maxPositionEmbeddings = 512;
		
		
		Transformer tf = new Transformer();
		
		tf.number = batchSize;
		
		CLIPTextEmbeddingLayer layer = new CLIPTextEmbeddingLayer(vocabSize, embedDim, maxPositionEmbeddings, tf);
		
		float[] data = RandomUtils.order(batchSize * time, 1f, 0f);
		
		Tensor input = new Tensor(batchSize , 1, 1, time, data, true);
		
		layer.forward(input);
		
		layer.getOutput().showShape();
		layer.getOutput().showDM();
		
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {

	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

	public EmbeddingIDLayer getPositionEmbedding() {
		return positionEmbedding;
	}

	public EmbeddingIDLayer getTokenEmbedding() {
		return tokenEmbedding;
	}

}
