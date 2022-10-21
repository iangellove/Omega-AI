package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.ConvKernel;
import com.omega.engine.gpu.DWeightKernel;
import com.omega.engine.gpu.DXKernel;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;

/**
 * 
 * ConvolutionLayer
 * 
 * @author Administrator
 *
 */
public class ConvolutionLayer extends Layer {

	public int kernelNum = 0;
	
	public int kWidth = 0;
	
	public int kHeight = 0;
	
	public int stride = 1;
	
	public int padding = 0;
	
	private ConvKernel convKernel;
	
	private DWeightKernel dWKernel;
	
	private DXKernel dXKernel;
	
	private float[] onceX;
	
	private float[] onceWX;
	
	private float[] onceDX;
	
	private float[] onceOut;
	
	private float[] onceDXOut;
	
	/**
	 * ConvolutionLayer
	 * @param channel
	 * @param kernelNum
	 * @param width
	 * @param height
	 * @param kWidth
	 * @param kHeight
	 * @param padding
	 * @param stride
	 * @param activeFunction
	 * @param updater
	 */
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,int stride) {
		this.kernelNum = kernelNum;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.initParam();
	}
	
	/**
	 * ConvolutionLayer
	 * @param channel
	 * @param kernelNum
	 * @param width
	 * @param height
	 * @param kWidth
	 * @param kHeight
	 * @param padding
	 * @param stride
	 * @param activeFunction
	 * @param updater
	 */
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,int stride,boolean hasBias) {
		this.kernelNum = kernelNum;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.hasBias = hasBias;
		this.initParam();
	}
	
	/**
	 * ConvolutionLayer
	 * @param channel
	 * @param kernelNum
	 * @param width
	 * @param height
	 * @param kWidth
	 * @param kHeight
	 * @param padding
	 * @param stride
	 * @param activeFunction
	 * @param updater
	 */
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,int stride,boolean hasBias,Network network) {
		this.kernelNum = kernelNum;
		this.channel = channel;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.hasBias = hasBias;
		this.network = network;
		this.initParam();
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
		
		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierReluRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaimingNormalRandom(kernelNum * channel * kHeight * kWidth, 0, kernelNum * kHeight * kWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.kHeight, this.kWidth * this.kHeight * this.kWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.heRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.oChannel * this.kHeight * this.kWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.val(kernelNum * channel * kHeight * kWidth, 0.1f));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth,RandomUtils.order(kernelNum * channel * kHeight * kWidth, 0.1f, 0.1f));
		this.bias = new Tensor(1, 1, 1, kernelNum);
		this.diffB = new Tensor(1, 1, 1, kernelNum);
		this.diffW = new Tensor(this.kernelNum, this.channel, this.kHeight, this.kWidth);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = new Tensor(number, oChannel, oHeight, oWidth);
		}
		if(convKernel == null){
			this.onceX = new float[channel * height * width];
			this.onceOut = new float[kernelNum * oHeight * oWidth];
			convKernel = new ConvKernel(this.index+"_conv", onceOut, channel, height, width, kernelNum, kHeight, kWidth, stride, padding);
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width);
		}
		if(dWKernel == null){
			this.onceWX = new float[channel * height * width];
			dWKernel = new DWeightKernel(this.index+"_conv_dw", diffW.data, channel, height, width, kernelNum, this.kHeight, this.kWidth, stride, padding);
		}
		if(dXKernel == null){
			this.onceDX = new float[kernelNum * oHeight * oWidth];
			this.onceDXOut = new float[channel * height * width];
			dXKernel = new DXKernel(this.index+"_conv_dx", onceDXOut, channel, height, width, kernelNum, this.kHeight, this.kWidth, this.padding, this.stride);
		}
		this.diffB = new Tensor(1, 1, 1, kernelNum);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
		convKernel.setKernel(this.weight.data);
		
		int onceLength = channel * height * width;
		
		int onceOutLength = oChannel * oHeight * oWidth;
		
		for(int n = 0;n<this.number;n++) {
			System.arraycopy(this.input.data, n * onceLength, this.onceX, 0, onceLength);
			convKernel.setX(onceX);
			convKernel.conv();
			System.arraycopy(convKernel.getOut(), 0, output.data, n * onceOutLength, onceOutLength);
		}

		if(this.hasBias) {
			for(int n = 0;n<this.number;n++) {
				for(int i = 0;i<onceOutLength;i++) {
					int c = i / (height * width);
					output.data[n * onceOutLength + i] += bias.data[c];
				}
			}
		}
		
//		System.out.println(JsonUtils.toJson(output.getByNumberAndChannel(0, 0)));
		
//		System.out.println(this.index+":"+(System.nanoTime() - start) / 1e6+"ms.");
	}

	/**
	 * delta = diff(i + 1) * f'(xi)
	 * dx = padding(delta) conv r180(kernel)
	 * dw = delta * px
	 * remark: px is zeropadding x
	 */
	@Override
	public void diff() {
		// TODO Auto-generated method stub

//		long start = System.nanoTime();
		
		/**
		 * 计算deltaW
		 */
		this.computeDeltaW();

		/**
		 * 计算deltaB
		 */
		if(this.hasBias) {
			
			int onceOutLength = oChannel * oHeight * oWidth;
			
			for(int n = 0;n<this.number;n++) {
				for(int i = 0;i<onceOutLength;i++) {
					int c = i / (height * width);
					diffB.data[c] += delta.data[n * onceOutLength + i] / number;
				}
			}
			
		}
		
		/**
		 * 计算diff
		 */
		if(PROPAGATE_DOWN) {
			
			/**
			 * dx = col2im(a)
			 * a = (weight)T * diff
			 * a[c * kh * kw * oh * ow]
			 * (weight)T[c * kh * kw * ko]
			 * diff[ko * oh * ow]
			 */
			dXKernel.setKernel(this.weight.data);
			
			int onceLength = kernelNum * oHeight * oWidth;
			
			for(int n = 0;n<this.number;n++) {
				System.arraycopy(delta.data, n * onceLength, onceDX, 0, onceLength);
				dXKernel.setDelta(onceDX);
				dXKernel.conv();
				System.arraycopy(dXKernel.getOut(), 0, diff.data, n * dXKernel.getOut().length, dXKernel.getOut().length);
			}

		}
		
//		System.out.println("back:"+(System.nanoTime() - start) / 1e6 + "ms.");
		
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
//		long start = System.nanoTime();

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		
//		System.out.println(JsonUtils.toJson(diffW.data));
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all back");
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
		if(this.updater != null){
			this.updater.updateForMatrix(this);
		}else{
			
			for(int i = 0;i<this.weight.getDataLength();i++) {
				this.weight.data[i] -= this.learnRate * this.diffW.data[i];
			}
			
			for(int i = 0;i<this.bias.getDataLength();i++) {
				this.bias.data[i] -= this.learnRate * this.diffB.data[i];
			}
			
		}
//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all update========>");
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.conv;
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
	public LayerInit save() {
		// TODO Auto-generated method stub
		return new ConvLayerInit(this);
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
	
	/**
	 * 计算deltaW
	 * 20220816: dw = diff * im2col(input)T 
	 * diff[knumber * oh * ow]
	 * im2col(input)T[oh * ow * C * kh * kw]
	 */
	public void computeDeltaW (){
		
		int onceXLength = channel * height * width;
		
		int  onceDiffLength = kernelNum * oHeight * oWidth;
		
		float[] onceDiff = new float[onceDiffLength];

		for(int n = 0;n<number;n++) {
			System.arraycopy(input.data, n * onceXLength, onceWX, 0, onceXLength);
			System.arraycopy(delta.data, n * onceDiffLength, onceDiff, 0, onceDiffLength);
			dWKernel.setX(onceWX);
			dWKernel.setKernel(onceDiff);
			dWKernel.conv();
		}
		
		diffW.data = dWKernel.getOut_D2H();

		dWKernel.clear();

		MatrixOperation.division_self(diffW.data, number);
		
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
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}
	
}
