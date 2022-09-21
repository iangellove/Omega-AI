package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.Col2imUtils;
import com.omega.common.utils.Dilation;
import com.omega.common.utils.Im2col4d2T;
import com.omega.common.utils.Im2colForWeight;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.Im2colUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.DWeightKernel;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.data.CacheDataSet;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.model.ConvLayerInit;
import com.omega.engine.nn.model.LayerInit;

import jcuda.jcublas.cublasOperation;

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
	
	public Tensor pInput1D;  //n * c * h * w im2col
	
	public Tensor dInput1D;
	
	public float[][][][] kernel;  //kn * c * kh * kw

	public float[] bias;
	
	public float[][][][] deltaW;
	
	public float[] deltaB;
	
	public float[][][][] dwd;
	
//	private float[] pi1d;
	
//	public float[] delta1d;
	
//	public float[][][][] pdwd;
	
	private DWeightKernel dWKernel;
	
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
//		this.kernel = RandomUtils.val(this.kernelNum, this.channel, this.kHeight, this.kWidth, 0.1f);
//		this.kernel = RandomUtils.order(this.kernelNum, this.channel, this.kHeight, this.kWidth, 0.1f, 0.1f);
		this.bias = MatrixUtils.zero(this.kernelNum);
//		this.diffPadding = (height + 2 * padding - kHeight) % stride;
		this.diffPadding = ((this.height - 1) * this.stride + this.kHeight - this.oHeight) / 2;
		this.deltaB = MatrixUtils.zero(this.kernelNum);
		this.deltaW = MatrixUtils.zero(this.kernelNum,this.channel,this.kHeight,this.kWidth);

		if(dWKernel == null){
			float[] diffW = new float[kernelNum * channel * kHeight * kWidth];
			dWKernel = new DWeightKernel(this.index+"_conv_dw", diffW, channel, height, width, kernelNum, this.kHeight, this.kWidth, stride, padding);
		}
		
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.pInput == null || this.number != this.pInput.number){
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
			this.pInput = Blobs.zero(number, channel, height + padding * 2, width + padding * 2, this.pInput);
			int pLength = this.channel * kHeight * kWidth * this.number * oHeight * oWidth;
			this.pInput1D = new Tensor(1, 1, 1, pLength);
//			this.pi1d = new float[number * channel * (height + padding * 2) * (width + padding * 2)];
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
		
		MatrixUtils.zero(this.deltaB);
		MatrixUtils.zero(this.deltaW);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
//		long start = System.nanoTime();
        
		MatrixOperation.zeroPadding(this.input.maxtir, this.pInput.maxtir, this.padding);

		Im2colToVector.im2col(this.pInput.maxtir, this.pInput1D.data, this.kHeight, this.kWidth, this.stride);

		MatrixOperation.convnVailByIm2ColGPUV2(this.pInput1D.data, this.kernel, this.number, this.channel, this.height + this.padding * 2, this.width + this.padding * 2, this.stride, this.output.maxtir);

		if(this.hasBias) {

			this.output.maxtir = MatrixOperation.add(this.output.maxtir, this.bias);
			
		}
		
//		System.out.println(index+":"+JsonUtils.toJson(MatrixUtils.transform(this.output.maxtir[1])));
		
//		System.out.println((System.nanoTime() - start) / 1e6+"ms.");
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
		
		int adj = (height + 2 * padding - kHeight) % stride;
		
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
		
		int ppi = kHeight - padding -1;
		
		if(this.stride > 1) {
			deltaP = MatrixOperation.zeroPadding(this.dwd, ppi, adj);
		}else {
			deltaP = MatrixOperation.zeroPadding(this.delta.maxtir, ppi);
		}
		
		/**
		 * 旋转kernel180
		 */
		float[][][][] kernel180 = MatrixOperation.rotate180V2(this.kernel);
		
		/**
		 * 计算当前层梯度
		 */
		MatrixOperation.convnDeltaByIm2ColGPUV2(deltaP, kernel180, this.diff.maxtir, 1);
		
//		float[] t1 = new float[number * channel * height * width];
//		
//		dx(delta.maxtir, kernel, t1);
//		
//		float[] t2 = MatrixUtils.transform(this.diff.maxtir);
//		
//		System.out.println(CheckArrayUtils.check(t1, t2));
//		
//		MatrixUtils.transform(t1, this.diff.maxtir, number, channel, height, width);
		
	}
	
	public void dx(float[][][][] delta,float[][][][] kernel,float[] diff){
		
		int m = channel * kHeight * kWidth;
		int n = oHeight * oWidth;
		int k = kernelNum;
		
		float[] kernelData = MatrixUtils.transform(kernel);
		
		float[] d = MatrixUtils.transform(delta);
		
		float[] donce = new float[kernelNum * oHeight * oWidth];
		
		float[] r = new float[m * n];
		
		float[] out = new float[channel * height * width];
		
		/**
		 * dx = col2im(a)
		 * a = (weight)T * diff
		 * a[c * kh * kw * oh * ow]
		 * (weight)T[c * kh * kw * ko]
		 * diff[ko * oh * ow]
		 */
		for(int ni = 0;ni<number;ni++) {
			
			System.arraycopy(d, ni * donce.length, donce, 0, donce.length);
			
			GPUOP.getInstance().multiplyFloat(m, n, k, kernelData, donce, r, 
					cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f, m, n, n);
			
			Col2imUtils col2im = new Col2imUtils(r, out, height, width, kHeight, kWidth, padding, stride, oHeight, oWidth, 0, out.length - 1);
			ForkJobEngine.run(col2im);
			
			System.arraycopy(out, 0, diff, ni * channel * height * width, channel * height * width);
		}
		
//		MatrixUtils.transform(all, diff, number, channel, height, width);
		
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
	 */
	public void computeDeltaW (){
		
		int ko = this.kernelNum;
		int kh = this.delta.height;
		int kw = this.delta.width;
		
		if(this.stride > 1) {
			kh = this.dwd[0][0].length;
			kw = this.dwd[0][0][0].length;
		}
		
		int oHeight = ((this.pInput.height - kh ) / 1) + 1;
		int oWidth = ((this.pInput.width - kw) / 1) + 1;
		
		int xm = this.channel * oHeight * oWidth;
		int xn = this.number * kh * kw;
		
		/**
		 * input im2col
		 */
		Im2colForWeight.im2col(this.pInput.maxtir, this.dInput1D.data, kh, kw, 1);

		float[] delta1d = null;

		if(this.stride == 1) {
			delta1d = Im2colUtils.kernalToVector(this.delta.maxtir, true);
		}else {
			Dilation.dilation(this.delta.maxtir, this.dwd, this.stride);
			delta1d = Im2colUtils.kernalToVector(this.dwd, true);
		}

		float[] c = MatrixUtils.zero(xm * ko);
		
		GPUOP.getInstance().multiplyFloat(xm, xn, ko, this.dInput1D.data, delta1d, c);
		
		Im2col4d2T.to4d(c, this.deltaW, this.kernelNum, this.channel, oHeight, oWidth, this.number);
		
//		
//		/**
//		 * test
//		 */
//		int onceXLength = channel * height * width;
//		
//		int  onceDiffLength = kernelNum * oHeight * oWidth;
//		
//		float[] onceDiff = new float[onceDiffLength];
//		
//		float[] onceWX = new float[channel * height * width];
//		
//		float[] delta1d_2 = MatrixUtils.transform(this.delta.maxtir);
//		
//		float[] input1d = MatrixUtils.transform(this.input.maxtir);
//		
//		for(int n = 0;n<number;n++) {
//			System.arraycopy(input1d, n * onceXLength, onceWX, 0, onceXLength);
//			System.arraycopy(delta1d_2, n * onceDiffLength, onceDiff, 0, onceDiffLength);
//			dWKernel.setX(onceWX);
//			dWKernel.setKernel(onceDiff);
//			dWKernel.conv();
//		}
//		
//		float[] tmp = dWKernel.getOut_D2H();
//		
//		MatrixOperation.division_self(tmp, number);
//		
////		System.out.println(CheckArrayUtils.check(tmp, MatrixUtils.transform(deltaW)));
////		System.out.println(deltaW[0][0][0][0]+"="+tmp[0]);
////		
//		dWKernel.clear();
//		
//		MatrixUtils.transform(tmp, deltaW, kernelNum, channel, kHeight, kWidth);
		
	}
	
}
