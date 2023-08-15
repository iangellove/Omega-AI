package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.cudnn.ConvCudnnKernel;
import com.omega.engine.nn.layer.gpu.BiasKernel;
import com.omega.engine.nn.layer.gpu.ConvBaseKernel;
import com.omega.engine.nn.layer.gpu.ConvKernel;
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
	
	private ConvBaseKernel kernel;
	
	private BiasKernel biasKernel;
	
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
		this.hasParams = true;
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
		this.hasParams = true;
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
		this.hasParams = true;
		this.initParam();
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		int dataLength = kernelNum * channel * kHeight * kWidth;
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaiming_normal(dataLength, this.oChannel * this.oHeight * this.oWidth, this.paramsInit), true);
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierReluRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth), true);
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.kaimingNormalRandom(kernelNum * channel * kHeight * kWidth, 0, kernelNum * kHeight * kWidth), true);
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.xavierRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.height * this.kHeight, this.kWidth * this.kHeight * this.kWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.heRandom(kernelNum * channel * kHeight * kWidth, this.channel * this.oChannel * this.kHeight * this.kWidth));
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.val(kernelNum * channel * kHeight * kWidth, 0.1f), true);
//		this.weight = new Tensor(kernelNum, channel, kHeight, kWidth, RandomUtils.order(kernelNum * channel * kHeight * kWidth, 0.1f, 0.01f), true);
		this.bias = new Tensor(1, 1, 1, kernelNum, true);
		this.diffB = new Tensor(1, 1, 1, kernelNum, true);
		this.diffW = new Tensor(this.kernelNum,this.channel,this.kHeight,this.kWidth, true);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}
		if(kernel == null){
			if(this.network.CUDNN) {
				kernel = new ConvCudnnKernel(this.network, channel, height, width, kernelNum, kHeight, kWidth, stride, padding);
			}else {
				kernel = new ConvKernel(channel, height, width, kernelNum, kHeight, kWidth, stride, padding);
			}
			if(this.hasBias) {
				biasKernel = new BiasKernel();
			} 
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width, true);
		}
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
//		weight.showDM(weight.dataLength-1);
		kernel.conv(input, weight, output);

		if(this.hasBias) {
			biasKernel.addConvBias(output, bias);
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
//		if(oWidth == 7) {
//			System.out.println(JsonUtils.toJson(delta.syncHost()));
//			
//		}
		
		/**
		 * 计算deltaW
		 * 20220816: dw = diff * im2col(input)T 
		 * diff[knumber * oh * ow]
		 * im2col(input)T[oh * ow * C * kh * kw]
		 */
		kernel.dw(input, delta, diffW);

		/**
		 * 计算deltaB
		 */
		if(this.hasBias) {
			biasKernel.backwardConvBias(diffB, delta);
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
			kernel.dx(delta, weight, diff);
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
//		System.out.println(this.index+":"+this.freeze);
		if(!this.freeze) {
			if(this.updater != null){
				this.updater.update(this);
			}else{
				
				for(int i = 0;i<this.weight.getDataLength();i++) {
					this.weight.data[i] -= this.learnRate * this.diffW.data[i];
				}
				
				for(int i = 0;i<this.bias.getDataLength();i++) {
					this.bias.data[i] -= this.learnRate * this.diffB.data[i];
				}
				
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
