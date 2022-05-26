package com.omega.engine.nn.layer;

import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;

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
	
	public float[][] pInput2D;  //n * c * h * w im2col
	
	public float[][][][] kernel;  //kn * c * kh * kw

	public float[] bias;
	
	public float[][][][] deltaW;
	
	public float[] deltaB;
	
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
			int stride) {
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
	public ConvolutionLayer(int channel,int kernelNum,int width,int height,int kWidth,int kHeight,int padding,
			int stride,boolean hasBias) {
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
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
//		this.kernel = MatrixOperation.gaussianRandom(this.channel, this.kernelNum, this.kHeight, this.kWidth, 0.01d);
//		this.kernel = MatrixOperation.heRandom(this.channel, this.kernelNum, this.kHeight, this.kWidth, this.width * this.height);
		this.kernel = RandomUtils.xavierRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.channel, this.oChannel);
		this.bias = MatrixUtils.zero(this.kernelNum);
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
//		System.out.println("{"+this.getLayerType()+index+"}oh:["+this.oHeight+"]k:["+this.kHeight +"]diffPadding:["+this.diffPadding+"]padding:["+this.padding+"]");
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
		this.deltaB = MatrixUtils.zero(this.kernelNum);
		this.deltaW = MatrixUtils.zero(this.kernelNum,this.channel,this.kHeight,this.kWidth);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		long start1 = System.nanoTime();
		this.pInput = Blobs.blob(MatrixOperation.zeroPadding(this.input.maxtir, this.padding),this.pInput);
//		System.out.println((System.nanoTime() - start1) / 1e6+"ms->zero-padding");
//		this.output.maxtir = MatrixOperation.convnVail(this.pInput.maxtir, this.kernel, this.stride);
		
//		this.output.maxtir = MatrixOperation.convnVailByIm2Col(this.pInput.maxtir, this.kernel, this.stride, false);
//		
//		System.out.println("");
//		System.out.println("================start=====================");
//		
//		PrintUtils.printImage(this.output.maxtir[0][0][0]);
//		long start2 = System.nanoTime();
		this.output.maxtir = MatrixOperation.convnVailByIm2ColGPU(this.pInput.maxtir, this.kernel, this.stride);
//		System.out.println((System.nanoTime() - start2) / 1e6+"ms->cov");
//		System.out.println("");
//		System.out.println("=================end====================");
//		
//		PrintUtils.printImage(this.output.maxtir[0][0][0]);

		
		if(this.hasBias) {
//			long start3 = System.nanoTime();
			this.output.maxtir = MatrixOperation.add(this.output.maxtir, this.bias);
//			System.out.println((System.nanoTime() - start3) / 1e6+"ms->add bias");
		}

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
//		System.out.println("");
//		System.out.println("diff:");
//		
//		MatrixOperation.printImage(this.delta.maxtir[0][0][0]);
//		System.out.println("==============diff");
		
		/**
		 * 计算deltaW
		 */
//		float[][][][] deltaWT = MatrixOperation.convnVailForDelta(this.pInput.maxtir, this.delta.maxtir, 1);
//		long start = System.nanoTime();
		/**
		 * input im2col
		 */
		float[][] pinput2d = Im2colUtils.im2col(this.pInput.maxtir, this.kernelNum, this.kHeight, this.kWidth, this.stride);
		
		float[][] pInputT = MatrixUtils.transpose(pinput2d);
		
		float[][] delta2d = Im2colUtils.to2d(this.delta.maxtir);
		
		float[][] dw2d = MatrixOperation.multiplicationByCuda(pInputT, delta2d);
		
		float[][][][] deltaWT = Im2colUtils.to4d(dw2d, this.kernelNum, this.channel, this.kHeight, this.kWidth);
		
		this.deltaW = MatrixOperation.division(deltaWT, this.number);
		
		if(this.hasBias) {
			
			this.deltaB = MatrixOperation.division(MatrixOperation.sumBias(this.delta.maxtir),this.number);
			
		}

		/**
		 * 梯度添加zeroPadding使得size与卷积输入一致
		 */
		float[][][][] deltaP = MatrixOperation.zeroPadding(this.delta.maxtir, this.diffPadding);

		/**
		 * 旋转kernel180
		 */
		float[][][][] kernel180 = MatrixOperation.rotate180V2(this.kernel);

		/**
		 * 计算当前层梯度
		 */
//		this.diff.maxtir = MatrixOperation.convnVailByIm2Col(deltaP, kernel180, 1, true);

		this.diff.maxtir = MatrixOperation.convnVailByIm2ColGPU(deltaP, kernel180, 1);

//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all back========>");
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
//		System.out.println((System.nanoTime() - start) / 1e6+"ms->all update========>");
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

	@Override
	public LayerInit save() {
		// TODO Auto-generated method stub
		return new ConvLayerInit(this);
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		
		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		float[][][][] pInput = MatrixOperation.zeroPadding(input, this.padding);
		
		output = MatrixOperation.convnVailByIm2Col(pInput, this.kernel, this.stride, false);

		output = MatrixOperation.add(output, this.bias);
		
		return output;
	}

}
