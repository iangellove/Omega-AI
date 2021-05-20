package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.updater.Updater;

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

	public int diffPadding = 0;
	
	public Blob pInput;  //n * c * h * w
	
	public double[][][][] kernel;  //c * kn * kh * kw

	public double[] bias;
	
	public double[][][][] deltaW;
	
	public double[] deltaB;
	
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
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,
			int stride,Updater updater) {
		this.channel = channel;
		this.kernelNum = kernelNum;
		this.width = width;
		this.height = height;
		this.kWidth = kWidth;
		this.kHeight = kHeight;
		this.padding = padding;
		this.stride = stride;
		this.updater = updater;
		this.initParam();
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
//		this.kernel = MatrixOperation.gaussianRandom(this.channel, this.kernelNum, this.kHeight, this.kWidth, 0.1d);
		this.kernel = MatrixOperation.heRandom(this.channel, this.kernelNum, this.kHeight, this.kWidth, this.width);
		this.bias = MatrixOperation.zero(this.kernelNum);
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
		this.deltaB = MatrixOperation.zero(this.oWidth);
		this.deltaW = MatrixOperation.zero(this.width,this.oWidth,this.kHeight,this.kWidth);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

		this.pInput = Blobs.blob(MatrixOperation.zeroPadding(this.input.maxtir, this.padding),this.pInput);
		
		this.output.maxtir = MatrixOperation.convnVail(this.pInput.maxtir, this.kernel, this.stride);
		
//		MatrixOperation.printImage(this.output[0]);
//		System.out.println("======================================>");
//		MatrixOperation.printImage(this.bias[0]);
		
		this.output.maxtir = MatrixOperation.add(this.output.maxtir, this.bias);
		
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
		
		/**
		 * 计算deltaW
		 */
		this.deltaW = MatrixOperation.division(MatrixOperation.convnVailForDelta(this.pInput.maxtir, this.delta.maxtir, 1), this.number);

		this.deltaB = MatrixOperation.division(MatrixOperation.sumBias(this.delta.maxtir),this.number);
		
		/**
		 * 梯度添加zeroPadding使得size与卷积输入一致
		 */
		double[][][][] deltaP = MatrixOperation.zeroPadding(this.delta.maxtir, this.diffPadding);
		
		/**
		 * 旋转kernel180
		 */
		double[][][][] kernel180 = MatrixOperation.rotate180(this.kernel);
		
//		System.out.println("----------------"+this.index+"-------------------------");
//		
//		MatrixOperation.printImage(this.nextDiff);
//		
//		System.out.println("==============="+this.index+"===================");
//		
//		MatrixOperation.printImage(deltaP);
		
		/**
		 * 计算当前层梯度
		 */
		this.diff.maxtir = MatrixOperation.convnVailForBack(kernel180, deltaP, 1);
		
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
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		if(this.updater != null){
			this.updater.updateForMatrix(this);
		}else{
			
			for(int c = 0;c<this.channel;c++) {
				for(int k = 0;k<this.kernelNum;k++) {
					for(int kh = 0;kh<this.kHeight;kh++) {
						for(int kw = 0;kw<this.kWidth;kw++) {
							this.kernel[c][k][kh][kw] -= this.learnRate * this.deltaW[c][k][kh][kw];
						}
					}
				}
			}
			
			for(int k = 0;k<this.kernelNum;k++) {
				for(int oh = 0;oh<this.oHeight;oh++) {
					for(int ow = 0;ow<this.oWidth;ow++) {
						this.bias[k] -= this.learnRate * this.deltaB[k];
					}
				}
			}
			
		}
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.conv;
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("conv layer["+this.index+"]diff start:");
//		
//		
//		System.out.println("conv layer["+this.index+"]diff end.");
	}

}
