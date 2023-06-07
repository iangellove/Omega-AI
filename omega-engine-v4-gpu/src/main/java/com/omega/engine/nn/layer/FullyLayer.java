package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.gpu.FullyKernel;

import jcuda.jcublas.cublasOperation;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class FullyLayer extends Layer{
	
	private FullyKernel kernel;
	
	public FullyLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasParams = true;
		this.initParam();
		initKernel();
	}

	public FullyLayer(int inputNum,int outputNum,boolean hasBias) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasBias = hasBias;
		this.hasParams = true;
		this.initParam();
		initKernel();
	}
	
	public void initKernel() {
		kernel = new FullyKernel();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width, true);
		}
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierReluRandom(this.width * this.oWidth, this.width, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaimingNormalRandom(this.width * this.oWidth, 0, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaimingUniformRandom(this.width * this.oWidth, 0, this.oWidth), true);
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.xavierRandom(this.width * this.oWidth, this.width, this.oWidth));
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.order(this.width * this.oWidth, 0.1f, 0.01f), true);
//		this.weight = new Tensor(1, 1, width, oWidth,RandomUtils.val(this.width * this.oWidth, 0.1f), true);
//		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.heRandom(this.width * this.oWidth, this.width * this.oWidth));
		this.bias = new Tensor(1, 1, 1, oWidth, true);
//		this.bias = new Tensor(1, 1, 1, oWidth, RandomUtils.kaimingUniformBias(x, n), true);
		this.diffB = new Tensor(1, 1, 1, oWidth, true);
		this.diffW = new Tensor(1, 1, width, oWidth, true);
		
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {

			GPUOP.getInstance().multiplyFloat(number, oWidth, width, input.getGpuData(), weight.getGpuData(), output.getGpuData(),
					cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
//			if(index == 7) {
////				input.showDM();
//				bias.showDM();
//			}
//			output.showDM();
//			System.out.println("---output---");
			
			if(hasBias) {
				kernel.addBias(output, bias);
			}
			
		}
		
	}
	
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
//		System.out.println("index-delta:"+index);
		
//		System.out.println(JsonUtils.toJson(delta.syncHost()));
		
//		delta.showDM();
		
		/**
		 * deltaW = inputT * delta
		 * int m,int n,int k, float A[],float B[], float C[],int CUBLAS_OP_A,int CUBLAS_OP_B,float alpha,float beta
		 * number * w
		 * number * ow
		 * m = w,k = number,n = ow
		 */
		GPUOP.getInstance().multiplyFloat(this.width, this.oWidth, this.number, input.getGpuData(), delta.getGpuData(), diffW.getGpuData(),
				cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);

		
		if(hasBias) {
			kernel.backwardBias(diffB, delta);
		}

		/**
		 * diff = delta * weightT
		 * number * ow
		 * w * ow
		 * m = number,k = ow,n = w
		 */
		GPUOP.getInstance().multiplyFloat(this.number, this.width, this.oWidth, delta.getGpuData(), weight.getGpuData(), diff.getGpuData(),
				cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 1.0f, 0.0f);
		
//		diff.showDM();
		
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
