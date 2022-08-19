package com.omega.engine.nn.layer;

import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.PoolingDiffKernel;
import com.omega.engine.gpu.PoolingKernel;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
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
	
	public float[][][][][] mask;
	
	private PoolingKernel pooling;
	
	private PoolingDiffKernel diffKernel;
	
	private float[] onceX;
	private float[] onceMask;
	private float[] onceOut;
	private float[] onceDiff;
	private float[] onceDelta;
	
	private float[] maskData;
	
	private float[] input1d;
	
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
		if(this.output == null || this.output.maxtir.length != number) {
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
			this.input1d = new float[number * channel * height * width];
			this.maskData = new float[number * channel * oHeight * oWidth * pHeight * pWidth];
		}
//		if(this.mask == null || this.mask.length != this.number){
//			this.mask = MatrixUtils.zero(this.number,this.channel, this.oHeight * this.oWidth, this.pHeight, this.pWidth);
//		}else {
//			MatrixUtils.zero(this.mask);
//		}
		
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
		if(diff == null || this.diff.maxtir.length != number) {
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
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
//		MatrixOperation.poolingAndMask(this.input.maxtir, this.mask, this.pWidth, this.pHeight, this.stride, this.poolingType, this.output.maxtir);
		
		Im2colUtils.kernalToVector2(this.input.maxtir, this.input1d, false);
		
		int onceLength = channel * height * width;
		
		for(int n = 0;n<number;n++) {
			System.arraycopy(this.input1d, n * onceLength, this.onceX, 0, onceLength);
			pooling.setX(onceX);
			pooling.pooling();
			System.arraycopy(pooling.getMask(), 0, this.maskData, n * onceMask.length, onceMask.length);
			MatrixUtils.col2im4d(pooling.getOut(), this.output.maxtir, n, channel, this.oHeight, this.oWidth);
		}
		
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
//		MatrixOperation.poolingDiff(this.delta.maxtir, this.mask, this.diff.maxtir, this.pWidth, this.pHeight, this.stride);
		
		float[] delta_v = Im2colUtils.kernalToVector2(this.delta.maxtir, false);
		
		for(int n = 0;n<number;n++) {
			System.arraycopy(delta_v, n * onceDelta.length, onceDelta, 0, onceDelta.length);
			System.arraycopy(maskData, n * onceMask.length, onceMask, 0, onceMask.length);
			diffKernel.setX(onceDelta);
			diffKernel.setMask(onceMask);
			diffKernel.diff();
			MatrixUtils.col2im4d(diffKernel.getOut(), this.diff.maxtir, n, channel, height, width);
		}
		
//		PrintUtils.printImage(this.diff.maxtir);
//		System.out.println("pooling layer ["+this.index+"]");
//		
//		MatrixOperation.printImage(this.nextDiff);
//		
//		System.out.println("pooling layer ["+this.index+"] end.");
		
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
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("pooling layer["+this.index+"]diff start:");
//		
//		MatrixOperation.printImage(this.diff);
//		
//		System.out.println("pooling layer["+this.index+"]diff end.");

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("pooling layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		float[][][][] output = MatrixOperation.poolingAndMask(input, this.mask,this.pWidth, this.pHeight, this.stride, this.poolingType);
		return output;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}
	
}
