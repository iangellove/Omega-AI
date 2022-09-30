package com.omega.engine.nn.layer.normalization;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BNKernel;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;

/**
 * 
 * Batch Normalization Layer
 * 
 * @author Administrator
 * 
 * mean = ∑x / m
 * std = (∑(x - mean)^2 / m)^1/2
 * zi = (xi - mean) / std
 * yi = gama * zi + beta
 */
public class BNLayer extends NormalizationLayer {
	
	public BNType bnType = null;

	/**
	 * if prelayer is conv layer meanNum = channel
	 * else if prelayer is fully layer meanNum = channel * height * width
	 */
	private int meanNum = 0;
	
	public float[] gama;
	
	public float[] beta;
	
	public float[] deltaGama;
	
	public float[] deltaBeta;
	
	public float[] runingMean;
	
	public float[] runingVar;
	
	private BNKernel kernel;
	
	public boolean hasRuning = true;
	
	public BNLayer() {
//		initParam();
	}
	
	public BNLayer(Network network) {
		this.network = network;
	}
	
	@Override
	public void init() {

		this.number = this.network.number;
		
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}

		if(this.bnType == null) {
			if(this.preLayer.getLayerType() == LayerType.conv) {
				this.setBnType(BNType.conv_bn);
				this.meanNum = this.channel;
			}else if(this.preLayer.getLayerType() == LayerType.full){
				this.setBnType(BNType.fully_bn);
				this.meanNum = this.channel * this.height * this.width;
			}
		}
		
		if(this.gama == null || this.beta == null) {
			this.gama = MatrixUtils.one(this.meanNum);
			this.beta = MatrixUtils.zero(this.meanNum);
			this.deltaGama = MatrixUtils.zero(this.meanNum);
			this.deltaBeta = MatrixUtils.zero(this.meanNum);
		}

		if(this.output == null) {
			this.diff = new Tensor(number, channel, height, width);
			this.output = new Tensor(number, oChannel, oHeight, oWidth);
		}
		
		if(this.number != this.output.number) {
			this.diff.number = number;
			this.output.number = number;
			this.diff.data = new float[number * channel * height * width];
			this.output.data = new float[number * channel * height * width];
		}
		
		if(getKernel() == null) {
			switch (this.getBnType()) {
			case fully_bn:
				kernel = new BNKernel(this.preLayer.index+":"+this.preLayer.getLayerType()+"_bn", this.getBnType(), output, diff, deltaGama, deltaBeta, number, width, height, channel);
				break;
			case conv_bn:
				kernel = new BNKernel(this.preLayer.index+":"+this.preLayer.getLayerType()+"_bn", this.getBnType(), output, diff, deltaGama, deltaBeta, number, channel, height, width);
				break;
			}
		}
		
		if(!hasRuning) {
			hasRuning = true;
			kernel.setRuningMean(runingMean);
			kernel.setRuningVar(runingVar);
		}
		
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void initBack() {
//		if(this.diff == null || this.number != this.diff.number) {
//			this.diff = new Tensor(number, oChannel, oHeight, oWidth);
//		}
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

		getKernel().setX(input, number);
		
		getKernel().setGama(gama, beta);
		
		getKernel().forward(this.network.RUN_MODEL);
		
	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
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

	/**
	 * 原论文公式
	 * deltaGama = ∑ deta * z
	 * deltaBeta = ∑ deta
	 * dxhat = deta * gama
	 * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
	 * dmean = ∑ dxhat * -1 / (var + eta)^1/2 + dvar * (∑ -2 * (x - mean)) / n
	 * dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
	 * darknet公式
	 * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
	 */
	@Override
	public void diff() {
		
//		long start = System.nanoTime();
		
		getKernel().setDelta(delta);
		
		getKernel().backward();
		
//		MatrixOperation.division_self(deltaGama, number);
//		
//		MatrixOperation.division_self(deltaBeta, number);
		
//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
		
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
		
//		this.diff_caffe();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		if(this.updater != null){
			this.updater.updateForBN(this);
		}else{
			for(int i = 0;i<this.gama.length;i++) {
				this.gama[i] -= this.learnRate * this.deltaGama[i];
			}
			for(int i = 0;i<this.beta.length;i++) {
				this.beta[i] -= this.learnRate * this.deltaBeta[i];
			}
		}
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.bn;
	}

	@Override
	public LayerInit save() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		
		return null;
	}
	
	public BNType getBnType() {
		return bnType;
	}

	public void setBnType(BNType bnType) {
		this.bnType = bnType;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}

	public BNKernel getKernel() {
		return kernel;
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
