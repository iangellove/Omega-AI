package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class BasicBlockLayer extends Layer {

	private ConvolutionLayer identityConv;
	
	private BNLayer identityBN;
	
	private ConvolutionLayer conv1;
	
	private BNLayer bn1;
	
	private ActiveFunctionLayer a1;
	
	private ConvolutionLayer conv2;
	
	private BNLayer bn2;

	private int kHeight = 3;
	
	private int kWidth = 3;
	
	private int padding = 1;
	
	private boolean downSample = false;
	
	private int fisrtLayerStride = 2;
	
	public BasicBlockLayer(int channel,int oChannel,int height,int width,boolean downSample, Network network) {
		
		this.network = network;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.downSample = downSample;
		
		if(this.isDownSample()) {
			this.oHeight = (height + padding * 2 - kHeight) / fisrtLayerStride + 1;
			this.oWidth = (width + padding * 2 - kWidth) / fisrtLayerStride + 1;
		}else {
			this.oHeight = height;
			this.oWidth = width;
		}

		initLayers();
		
	}
	
	public void initLayers() {
		
		if(isDownSample()) {

			identityConv = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, fisrtLayerStride, false, this.network);
			identityConv.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
			
			identityBN = new BNLayer(this.network);
			identityBN.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
			identityBN.setPreLayer(identityConv);

			conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, fisrtLayerStride, false, this.network);
		}else {
			conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, false, this.network);
		}
		
		conv1.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
		
		bn1 = new BNLayer(this.network);
		bn1.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
		bn1.setPreLayer(conv1);
		
		a1 = new ReluLayer(this.network);
		a1.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
		a1.setPreLayer(bn1);
		
		conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false, this.network);
		conv2.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
		
		bn2 = new BNLayer(this.network);
		bn2.setUpdater(UpdaterFactory.create(this.network.updater, this.network));
		bn2.setPreLayer(conv2);
		
	}

	@Override
	public void init() {
		this.number = this.network.number;
		if(this.output == null || this.output.number != this.network.number) {
			this.output = new Tensor(number, oChannel, oHeight, oWidth);
		}
	}
	
	@Override
	public void initBack() {
		if(this.diff == null || this.diff.number != this.network.number) {
			this.diff = new Tensor(number, channel, height, width);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		float[] x = this.input.data;
		
		if(isDownSample()) {
			identityConv.forward(this.input);
			identityBN.forward(identityConv.output);			
			x = identityBN.output.data;
		}
		
		conv1.forward(this.input);
		bn1.forward(conv1.output);
		a1.forward(bn1.output);
		
		conv2.forward(a1.output);
		bn2.forward(conv2.output);
		
		float[] x2 = bn2.output.data;

		/**
		 * if downSample 
		 * o = identity(x) + f(x)
		 * else
		 * o = x + f(x)
		 */
		this.output.data = MatrixOperation.add(x, x2);
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * deltax = deltao * (f'(x) + 1)
		 */
		bn2.back(delta);
		conv2.back(bn2.diff);
		
		a1.back(conv2.diff);
		bn1.back(a1.diff);
		conv1.back(bn1.diff);
		
		if(isDownSample()) {
			
			identityBN.back(delta);
			identityConv.back(identityBN.diff);
			
			this.diff.data = MatrixOperation.add(conv1.diff.data, identityConv.diff.data);
			
		}else {
			this.diff.data = MatrixOperation.add(conv1.diff.data, this.delta.data);
		}

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

		if(isDownSample()) {
			identityConv.update();
			identityBN.update();
		}
		
		conv1.update();
		bn1.update();
		a1.update();
		conv2.update();
		bn2.update();
		
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

	public boolean isDownSample() {
		return downSample;
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

}
