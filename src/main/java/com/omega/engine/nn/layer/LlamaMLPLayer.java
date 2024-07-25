package com.omega.engine.nn.layer;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * PoswiseFeedForward Layer
 * @author Administrator
 *
 */
public class LlamaMLPLayer extends Layer{
	
	private int embedDim = 0;
	
	private int nChannel = 1;
	
	private boolean bias = false;
	
	private boolean dropout = false;
	
	private FullyLayer linear1;
	private SiLULayer active;
	private FullyLayer linear2;
	
	private FullyLayer linear3;

	private DropoutLayer dropoutLayer;
	
	private Tensor tmp;
	
	private int multiple_of = 32;
	
	public LlamaMLPLayer(int embedDim,int nChannel,boolean bias) {
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.nChannel = 4 * embedDim;
		this.nChannel = (int)(2 * nChannel / 3);
		this.nChannel = multiple_of * ((this.nChannel + multiple_of - 1) / multiple_of);
		this.initLayers();
	}
	
	public LlamaMLPLayer(int embedDim,int nChannel,boolean bias,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.embedDim = embedDim;
		this.nChannel = nChannel;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.nChannel = 4 * embedDim;
		this.nChannel = (int)(2 * nChannel / 3);
		this.nChannel = multiple_of * ((this.nChannel + multiple_of - 1) / multiple_of);
		this.initLayers();
	}
	
	public void initLayers() {
//		NanoGPT net = (NanoGPT) this.network;
		this.linear1 = new FullyLayer(embedDim, nChannel, bias, network);
//		this.linear1.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.01f, 0.02f), true);
//		this.linear1.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.order(this.embedDim * nChannel, 0.001f, 0.001f), true);
//		Tensor qw = new Tensor(1, 1, embedDim, nChannel, true);
//		TensorOP.permute(this.linear1.weight, qw, new int[] {0, 1, 3, 2});
//		this.linear1.weight = qw;
//		this.linear1.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, 0.02f), true);

		this.active = new SiLULayer(linear1);
		
		this.linear2 = new FullyLayer(nChannel, embedDim, bias, network);
//		this.linear2.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.uniform(this.nChannel * embedDim, 0.01f, 0.02f), true);
//		this.linear2.weight = new Tensor(1, 1, embedDim, nChannel, RandomUtils.order(this.nChannel * embedDim, 0.001f, 0.001f), true);
//		Tensor w2 = new Tensor(1, 1, nChannel, embedDim, true);
//		TensorOP.permute(this.linear2.weight, w2, new int[] {0, 1, 3, 2});
//		this.linear2.weight = w2;
//		this.linear2.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, 0.02f), true);
//		this.linear2.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.0f, (0.02f / (float) Math.sqrt(2 * net.decoderNum))), true);
		
		this.linear3 = new FullyLayer(embedDim, nChannel, bias, network);
//		this.linear3.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.uniform(this.embedDim * nChannel, 0.01f, 0.02f), true);
//		this.linear3.weight = new Tensor(1, 1, nChannel, embedDim, RandomUtils.order(this.embedDim * nChannel, 0.001f, 0.001f), true);
//		Tensor w3 = new Tensor(1, 1, embedDim, nChannel, true);
//		TensorOP.permute(this.linear3.weight, w3, new int[] {0, 1, 3, 2});
//		this.linear3.weight = w3;
		
		if(dropout) {
			dropoutLayer = new DropoutLayer(0.1f, linear2);
		}
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		if(this.tmp == null || this.number != this.tmp.number){
			this.tmp = Tensor.createGPUTensor(this.tmp, number, oChannel, oHeight, nChannel, true);
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
		
		linear1.forward(input);
		
		active.forward(linear1.getOutput());
		
		linear3.forward(input);
		
		TensorOP.mul(active.getOutput(), linear3.getOutput(), tmp);
		
		linear2.forward(tmp);
		
		if(dropout) {
			dropoutLayer.forward(linear2.getOutput());
			this.output = dropoutLayer.getOutput();
		}else {
			this.output = linear2.getOutput();
		}

	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		if(this.dropout) {
			this.dropoutLayer.back(delta);
			this.linear2.back(this.dropoutLayer.diff);
		}else {
			this.linear2.back(this.delta);
		}
		
		//diff l3
		TensorOP.mul(this.linear2.diff, active.getOutput(), tmp);
		linear3.back(tmp);
		
		//diff l1
		TensorOP.mul(this.linear2.diff, linear3.getOutput(), tmp);
		
		active.back(tmp);
		
		linear1.back(active.diff);
		
		TensorOP.add(this.linear1.diff, this.linear3.diff, this.linear1.diff);
		
		this.diff = this.linear1.diff;
		
//		System.out.println("mlp diff:");
//		diff.showDMByNumber(0);
		
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
		linear1.update();
		linear2.update();
		linear3.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.feed_forward;
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
		
	}
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		linear1.saveModel(outputStream);
		linear3.saveModel(outputStream);
		linear2.saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		linear1.loadModel(inputStream);
		linear3.loadModel(inputStream);
		linear2.loadModel(inputStream);
	}
	
}
