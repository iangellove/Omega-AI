package com.omega.engine.nn.layer.vqvae.tiny;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.gpu.BasicBlockKernel;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class VQVAEResidual extends Layer {

	private int group = 32;
	
	private BasicBlockKernel kernel;
	
	private GNLayer norm1;
	
	private ActiveFunctionLayer a1;
	
	private ConvolutionLayer conv1;
	
	private GNLayer norm2;
	
	private ActiveFunctionLayer a2;
	
	private ConvolutionLayer conv2;

	private ConvolutionLayer conv_shortcut;
	
	private boolean shortcut = false;
	
	public VQVAEResidual(int channel,int oChannel,int height,int width,int group, Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = channel;
		this.height = height;
		this.width = width;
		this.group = group;
		
		if(channel != oChannel) {
			shortcut = true;
		}
		
		kernel = new BasicBlockKernel();
		
		initLayers(oChannel);
		
		this.oHeight = conv2.oHeight;
		this.oWidth = conv2.oWidth;
		this.oChannel = conv2.oChannel;
	}
	
	public void initLayers(int oChannel) {
		
		norm1 = new GNLayer(group, channel, height, width, BNType.conv_bn, this);
		a1 = new SiLULayer(norm1);
		
		conv1 = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, true, this.network);
		conv1.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
		conv1.paramsInit = ParamsInit.silu;
		
		norm2 = new GNLayer(group, conv1, BNType.conv_bn);
		a2 = new SiLULayer(norm2);
		
		conv2 = new ConvolutionLayer(conv1.oChannel, oChannel, conv1.oWidth, conv1.oHeight, 3, 3, 1, 1, false, this.network);
		conv2.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
		conv2.paramsInit = ParamsInit.silu;

		if(shortcut) {
			conv_shortcut = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, false, this.network); 
			conv_shortcut.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
			conv_shortcut.paramsInit = ParamsInit.silu;
		}

	}

	@Override
	public void init() {
		this.number = this.network.number;
		if(this.output == null || this.output.number != this.network.number) {
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	@Override
	public void initBack() {
//		if(this.cache_delta == null || this.output.number != cache_delta.number){
//			this.cache_delta = Tensor.createGPUTensor(this.cache_delta, number, oChannel, oHeight, oWidth, true);
//		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		norm1.forward(this.input);
		a1.forward(norm1.getOutput());
		conv1.forward(a1.getOutput());
		norm2.forward(conv1.getOutput());
		a2.forward(norm2.getOutput());
		conv2.forward(a2.getOutput());
		
		if(shortcut) {
			conv_shortcut.forward(this.input);
			kernel.add(conv_shortcut.getOutput(), conv2.getOutput(), output);
		}else {
			kernel.add(input, conv2.getOutput(), output);
		}
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		System.out.println(index);
		/**
		 * deltax = deltao * (f'(x) + 1)
		 */
//		baseKernel.copy_gpu(delta, this.cache_delta, delta.getDataLength(), 1, 1);

		conv2.back(delta);
		a2.back(conv2.diff);
		norm2.back(a2.diff);
		
		conv1.back(norm2.diff);
		a1.back(conv1.diff);
		norm1.back(a1.diff);

		if(shortcut) {
			conv_shortcut.back(this.delta);
			kernel.add(norm1.diff, conv_shortcut.diff, conv1.diff);
		}else {
			kernel.add(norm1.diff, this.delta, conv1.diff);
		}
		
		this.diff = conv1.diff;
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
		
		norm1.update();
		conv1.update();

		norm2.update();
		conv2.update();
		
		if(shortcut) {
			conv_shortcut.update();
		}
		
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		
		norm1.saveModel(outputStream);
//		norm1.gamma.showDM("down-res-norm1");
		conv1.saveModel(outputStream);
		norm2.saveModel(outputStream);
		conv2.saveModel(outputStream);
		
		if(shortcut) {
			conv_shortcut.saveModel(outputStream);
		}

	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		
		norm1.loadModel(inputStream);
//		norm1.gamma.showDM("down-res-norm1");
		conv1.loadModel(inputStream);
		norm2.loadModel(inputStream);
//		norm2.gamma.showDM("down-res-norm2");
		conv2.loadModel(inputStream);
		
		if(shortcut) {
			conv_shortcut.loadModel(inputStream);
		}
		
	}

}
