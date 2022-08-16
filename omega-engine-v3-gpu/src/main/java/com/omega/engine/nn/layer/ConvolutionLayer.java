package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.ConvKernel;
import com.omega.engine.gpu.ConvWeightKernel;
import com.omega.engine.gpu.data.CacheDataSet;
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
	
//	public Tensor pInput1D;  //n * c * h * w im2col
	
	public Tensor dInput1D;
	
	public float[][][][] kernel;  //kn * c * kh * kw

	public float[] bias;
	
	public float[][][][] deltaW;
	
	public float[] deltaB;
	
	public float[][][][] dwd;
	
	private float[] pi1d;
	
//	public float[] delta1d;
	
//	public float[][][][] pdwd;
	
	private ConvKernel convKernel;
	
	private ConvWeightKernel detalWKernel;
	
	private ConvKernel detalXKernel;
	
	private float[] onceX;
	
	private float[] onceWX;
	
	private float[] onceDX;
	
	private float[] onceOut;
	
	private float[] onceDWOut;
	
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
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.oChannel = this.kernelNum;
		this.oWidth = (this.width + this.padding * 2 - kWidth) / this.stride + 1;
		this.oHeight = (this.height + this.padding * 2 - kHeight) / this.stride + 1;
//		this.kernel = MatrixOperation.gaussianRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, 0.01d);
//		this.kernel = RandomUtils.heRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.width * this.height);
		this.kernel = RandomUtils.xavierRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.channel * this.height * this.width, this.oChannel * this.oHeight * this.oWidth);
//		this.kernel = RandomUtils.heRandom(this.kernelNum, this.channel, this.kHeight, this.kWidth, this.channel * this.oChannel * this.height * this.width);
		this.bias = MatrixUtils.zero(this.kernelNum);
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
		this.deltaB = MatrixUtils.zero(this.kernelNum);
		this.deltaW = MatrixUtils.zero(this.kernelNum,this.channel,this.kHeight,this.kWidth);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.pInput == null || this.number != this.pInput.number){
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
			this.pInput = Blobs.zero(number, channel, height + padding * 2, width + padding * 2, this.pInput);
//			int pLength = this.channel * kHeight * kWidth * this.number * oHeight * oWidth;
//			this.pInput1D = new Tensor(1, 1, 1, pLength);
			this.pi1d = new float[number * channel * (height + padding * 2) * (width + padding * 2)];
		}
		if(convKernel == null){
			this.onceX = new float[channel * (height + this.padding * 2) * (width + this.padding * 2)];
			this.onceOut = new float[kernelNum * oHeight * oWidth];
			convKernel = new ConvKernel(this.index+"_conv", onceOut, channel, height + this.padding * 2, width + this.padding * 2, kernelNum, kHeight, kWidth, stride);
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
			int kh = this.oHeight;
			int kw = this.oWidth;
			if(this.stride > 1) {
				kh = this.oHeight + (this.oHeight - 1) * (this.stride - 1);
				kw = this.oWidth + (this.oWidth - 1) * (this.stride - 1);
				this.dwd = new float[this.number][this.oChannel][kh][kw];
			}
			int oHeight = ((this.pInput.height - kh ) / 1) + 1;
			int oWidth = ((this.pInput.width - kw) / 1) + 1;
			int xm = this.channel * oHeight * oWidth;
			int xn = this.number * kh * kw;
			int pLength = xm * xn;
			this.dInput1D = new Tensor(1, 1, 1, pLength);
//			this.delta1d = new float[this.number * this.oChannel * kh * kw];
		}
		if(detalWKernel == null){
			this.onceWX = new float[channel * (height + this.padding * 2) * (width + this.padding * 2)];
			this.onceDWOut = new float[kernelNum * channel * kHeight * kWidth];
			detalWKernel = new ConvWeightKernel(this.index+"_conv_dw", onceDWOut, channel, height + this.padding * 2, width + this.padding * 2, kernelNum, this.kHeight, this.kWidth, stride);
		}
		if(detalXKernel == null){
			int ih = this.diff.height + diffPadding * 2;
			int iw = this.diff.width + diffPadding * 2;
			this.onceDX = new float[kernelNum * ih * iw];
			this.onceDXOut = new float[channel * height * width];
			detalXKernel = new ConvKernel(this.index+"_conv_dx", onceDXOut, kernelNum, ih, iw, channel, this.kHeight, this.kWidth, 1);
		}
		MatrixUtils.zero(this.deltaB);
		MatrixUtils.zero(this.deltaW);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
//		
		MatrixOperation.zeroPadding(this.input.maxtir, this.pInput.maxtir, this.padding);
		
		this.pi1d = MatrixUtils.transform(this.pInput.maxtir);

		/**
		 * kernel im2col
		 */
		float[] ka = Im2colUtils.kernalToVector2(this.kernel, false);
		
		convKernel.setKernel(ka);
		
		int onceLength = channel * (height + this.padding * 2) * (width + this.padding * 2);
		
		for(int n = 0;n<this.number;n++) {
			System.arraycopy(this.pi1d, n * onceLength, this.onceX, 0, onceLength);
			convKernel.setX(onceX);
			convKernel.conv();
			MatrixUtils.col2im4d(convKernel.getOut(), this.output.maxtir, n, this.kernelNum, this.oHeight, this.oWidth);
		}

		if(this.hasBias) {

			this.output.maxtir = MatrixOperation.add(this.output.maxtir, this.bias);

		}
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

		if(this.hasBias) {
			this.deltaB = MatrixOperation.division(MatrixOperation.sumBias(this.delta.maxtir),this.number);
		}

		/**
		 * 梯度添加zeroPadding使得size与卷积输入一致
		 */
		float[][][][] deltaP = null;
		
		if(this.stride > 1) {
			deltaP = MatrixOperation.zeroPadding(this.dwd, this.stride);
		}else {
			deltaP = MatrixOperation.zeroPadding(this.delta.maxtir, this.diffPadding);
		}
		
		float[] ix = MatrixUtils.transform(deltaP);
		
		/**
		 * 旋转kernel180
		 */
		float[][][][] kernel180 = MatrixOperation.rotate180V2(this.kernel);
		
		float[] ka = Im2colUtils.kernalToVector2(kernel180, true);

		detalXKernel.setKernel(ka);
		
		int onceLength = kernelNum * deltaP[0][0].length * deltaP[0][0][0].length;

		for(int n = 0;n<this.number;n++) {
			System.arraycopy(ix, n * onceLength, this.onceDX, 0, onceLength);
			detalXKernel.setX(onceDX);
			detalXKernel.conv();
			MatrixUtils.col2im4d(detalXKernel.getOut(), this.diff.maxtir, n, this.channel, this.height, this.width);
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
//		long start = System.nanoTime();
//		System.out.println("back start.");
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

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("conv layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
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

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
		CacheDataSet cache = new CacheDataSet(this.number);
		
		this.setTampDataSet(cache);
		
		/**
		 * 创建输出层矩阵乘法缓存
		 */
		float[] r = new float[this.number * this.kernelNum * oHeight * oWidth];
		cache.getDim1dSet().add(r);
		
		/**
		 * 创建输入层im2col缓存
		 */
		float[][] col = new float[this.number * oHeight * oWidth][this.kHeight * this.kWidth * this.channel];
		cache.getDim2dSet().add(col);
		
		float[] col2 =  new float[this.number * oHeight * oWidth * this.kHeight * this.kWidth * this.channel];
		cache.getDim1dSet().add(col2);
	}
	
	/**
	 * 计算deltaW
	 * 20220816: dw = diff * im2col(input)T 
	 * diff[knumber * oh * ow]
	 * im2col(input)T[oh * ow * C * kh * kw]
	 */
	public void computeDeltaW (){
//		
//		int ko = this.kernelNum;
//		int kh = this.delta.height;
//		int kw = this.delta.width;
//		
//		if(this.stride > 1) {
//			kh = this.dwd[0][0].length;
//			kw = this.dwd[0][0][0].length;
//		}
//		
//		int oHeight = ((this.pInput.height - kh ) / 1) + 1;
//		int oWidth = ((this.pInput.width - kw) / 1) + 1;
//		
//		int xm = this.channel * oHeight * oWidth;
//		int xn = this.number * kh * kw;
//		
//		
		
		float[] ka = Im2colUtils.kernalToVector2(this.delta.maxtir, false);
		
		int onceXLength = channel * (height + this.padding * 2) * (width + this.padding * 2);
		
		int  onceDiffLength = kernelNum * oHeight * oWidth;
		
		float[] onceDiff = new float[onceDiffLength];
//		long start = System.nanoTime();
		for(int n = 0;n<number;n++) {
			System.arraycopy(this.pi1d, n * onceXLength, onceWX, 0, onceXLength);
			System.arraycopy(ka, n * onceDiffLength, onceDiff, 0, onceDiffLength);
			detalWKernel.setX(onceWX);
			detalWKernel.setKernel(onceDiff);
			detalWKernel.conv();
		}

		MatrixUtils.col2im4dWeight(detalWKernel.getOut(), deltaW, kernelNum, channel, kHeight, kWidth, number);
		detalWKernel.clear();
//		System.out.println((System.nanoTime() - start) / 1e6+"ms.");
//		System.out.println((System.nanoTime() - start) / 1e6+"ms.");
//
//		Im2colForWeight.im2col(this.pInput.maxtir, this.dInput1D.data, kh, kw, 1);
//
//		float[] delta1d = null;
//
//		if(this.stride == 1) {
//			delta1d = Im2colUtils.kernalToVector(this.delta.maxtir, true);
//		}else {
//			Dilation.dilation(this.delta.maxtir, this.dwd, this.stride);
//			delta1d = Im2colUtils.kernalToVector(this.dwd, true);
//		}
//
//		float[] c = MatrixUtils.zero(xm * ko);
//		
//		GPUOP.getInstance().multiplyFloat(xm, xn, ko, this.dInput1D.data, delta1d, c);
//		
//		Im2col4d2T.to4d(c, this.deltaW, this.kernelNum, this.channel, oHeight, oWidth, this.number);
//		
	}
	
}
