package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.PoolingDiffKernel;
import com.omega.engine.gpu.PoolingKernel;
import com.omega.engine.pooling.PoolingType;

/**
 * PoolingLayer
 * 
 * @author Administrator
 *
 */
public class PoolingLayer extends Layer {
	
	public PoolingType poolingType;
	
	public int pWidth = 0;
	
	public int pHeight = 0;
	
	public int stride = 1;
	
	public float[] mask;
	
	private PoolingKernel pooling;
	
	private PoolingDiffKernel diffKernel;
	
	private float[] onceX;
	private float[] onceMask;
	private float[] onceOut;
	private float[] onceDiff;
	private float[] onceDelta;
	
	public PoolingLayer(int channel,int width,int height,int pWidth,int pHeight,int stride,PoolingType poolingType) {
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.pWidth = pWidth;
		this.pHeight = pHeight;
		this.stride = stride;
		this.poolingType = poolingType;
		initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.output.number != number) {
			this.output = new Tensor(number, oChannel, oHeight, oWidth);
			this.mask = new float[number * channel * oHeight * oWidth * pHeight * pWidth];
		}

		if(pooling == null) {
			onceX = new float[channel * height * width];
			onceMask = new float[channel * oHeight * oWidth * pHeight * pWidth];
			onceOut = new float[channel * oHeight * oWidth];
			pooling = new PoolingKernel(poolingType, onceOut, onceMask, channel, height, width, pHeight, pWidth, stride);
		}
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(diff == null || this.diff.number != number) {
			this.diff = new Tensor(number, channel, height, width);
		}
		
		if(diffKernel == null) {
			onceDiff = new float[channel * height * width];
			onceDelta = new float[channel * oHeight * oWidth];
			diffKernel = new PoolingDiffKernel(poolingType, onceDiff, channel, height, width, pHeight, pWidth, stride);
		}
		
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.channel;
		this.oWidth = (this.width - pWidth) / this.stride + 1;
		this.oHeight = (this.height - pHeight) / this.stride + 1;
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

		int onceLength = channel * height * width;
		int onceOutLength = oChannel * oHeight * oWidth;
		
		for(int n = 0;n<number;n++) {
			System.arraycopy(this.input.data, n * onceLength, this.onceX, 0, onceLength);
			pooling.setX(onceX);
			pooling.pooling();
			System.arraycopy(pooling.getMask(), 0, this.mask, n * onceMask.length, onceMask.length);
			System.arraycopy(pooling.getOut(), 0, this.output.data, n * onceOutLength, onceOutLength);
		}
		
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		int onceLength = channel * height * width;
		for(int n = 0;n<number;n++) {
			System.arraycopy(this.delta.data, n * onceDelta.length, onceDelta, 0, onceDelta.length);
			System.arraycopy(mask, n * onceMask.length, onceMask, 0, onceMask.length);
			diffKernel.setX(onceDelta);
			diffKernel.setMask(onceMask);
			diffKernel.diff();
			System.arraycopy(diffKernel.getOut(), 0, this.diff.data, n * onceLength, onceLength);
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
	public void update() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.pooling;
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

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
	
}
