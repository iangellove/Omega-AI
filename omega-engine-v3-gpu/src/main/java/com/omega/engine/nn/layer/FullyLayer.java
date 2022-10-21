package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;

import jcuda.jcublas.cublasOperation;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{
	
	public FullyLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.initParam();
	}

	public FullyLayer(int inputNum,int outputNum,boolean hasBias) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.initParam();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width);
		}
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = new Tensor(number, oChannel, oHeight, oWidth);
		}
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierReluRandom(this.width * this.oWidth, this.width, this.oWidth));
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaimingNormalRandom(this.width * this.oWidth, 0, this.oWidth));
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierRandom(this.width * this.oWidth, this.width, this.oWidth));
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.order(this.width * this.oWidth, 0.1f, 0.1f));
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.val(this.width * this.oWidth, 0.1f));
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.heRandom(this.width * this.oWidth, this.width * this.oWidth));
		this.bias = new Tensor(1, 1, 1, oWidth);
		this.diffB = new Tensor(1, 1, 1, oWidth);
		this.diffW = new Tensor(1, 1, width, oWidth);
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {
			
//			System.out.println("full-input:"+JsonUtils.toJson(input.data));
			
			GPUOP.getInstance().multiplyFloat(this.number, this.oWidth, this.width, input.data, weight.data, output.data, 
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f, this.width, this.oWidth, this.oWidth);

			if(hasBias) {

				for(int n = 0;n<this.number;n++) {
					for(int ow = 0;ow<this.oWidth;ow++) {
						output.data[n * oWidth + ow] += bias.data[ow];
					}
				}
			
			}
			
		}
		
	}
	
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
//		System.out.println("index-delta:"+index);
		
		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPUOP.getInstance().multiplyFloat(this.width, this.oWidth, this.number, input.data, delta.data, diffW.data,
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f, this.width, this.oWidth, this.oWidth);

		MatrixOperation.multiplication_self(diffW.data, (1.0f / this.number));
		
		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPUOP.getInstance().multiplyFloat(this.number, this.width, this.oWidth, delta.data, weight.data, diff.data,
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f, this.oWidth, this.oWidth, this.width);
		
		if(hasBias) {
			
			for(int ow = 0;ow<this.oWidth;ow++) {
				diffB.data[ow] = 0.0f;
				for(int n = 0;n<this.number;n++) {
					diffB.data[ow] += delta.data[n * oWidth + ow] / number;
				}
			}
		
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

	/**
	 * w(t) = w(t-1) + θ * deltaW
	 * b(t) = b(t-1) + θ * deltaB
	 * θ : learningRate
	 */
	@Override
	public void update() {
		// TODO Auto-generated method stub
		
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
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.full;
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
