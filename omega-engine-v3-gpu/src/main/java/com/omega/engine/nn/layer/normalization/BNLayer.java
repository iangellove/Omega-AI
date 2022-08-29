package com.omega.engine.nn.layer.normalization;

import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BNKernel;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.model.LayerInit;

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
	
	public float[] mean;
	
	public float[] var;

	/**
	 * if prelayer is conv layer meanNum = channel
	 * else if prelayer is fully layer meanNum = channel * height * width
	 */
	private int meanNum = 0;
	
	public float[] gama;
	
	public float[] beta;
	
	/**
	 * zi = (xi - mean) / (std + eta)^1/2
	 */
	public Blob z;
	
	public float[] deltaGama;
	
	public float[] deltaBeta;
	
	private BNKernel kernel;
	
	private float[] x;
	
	private float[] out;
	
	private float[] delta_v;
	
	private float[] x_diff;
	
	public BNLayer() {
//		initParam();
	}
	
	@Override
	public void init() {

		this.number = this.network.number;
		
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			if(this.parent != null) {
				preLayer = this.parent.layers.get(index - 1);
			}
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

		if(this.output == null || this.number != this.output.number) {
			this.out = new float[number * channel * height * width];
			this.delta_v = new float[number * channel * height * width];
			this.x_diff = new float[number * channel * height * width];
			this.x = new float[number * channel * height * width];
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		}
		
		if(kernel == null) {

			switch (this.getBnType()) {
			case fully_bn:
				kernel = new BNKernel(this.preLayer.index+":"+this.preLayer.getLayerType()+"_bn",out, x_diff, deltaGama, deltaBeta, number, width, height, channel);
				break;
			case conv_bn:
				kernel = new BNKernel(this.preLayer.index+":"+this.preLayer.getLayerType()+"_bn",out, x_diff, deltaGama, deltaBeta, number, channel, height, width);
				break;
			}
			
		}
		
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void initBack() {
		if(this.diff == null || this.number != this.diff.number) {
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
		}
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

		MatrixUtils.transform(this.input.maxtir,x);

		kernel.setX(x, number);
		
		kernel.setGama(gama, beta);
		
		kernel.forward(this.network.RUN_MODEL);
		
		MatrixUtils.transform(kernel.getOut(), this.output.maxtir, number, channel, height, width);

	}
	
	@Override
	public Blob getOutput() {
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
		
		MatrixUtils.transform(delta.maxtir, delta_v);
		
		kernel.setDelta(delta_v);
		
		kernel.backward();
		
		MatrixUtils.transform(kernel.getDiff(),this.diff.maxtir,number,channel,height,width);
		
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

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("bn layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
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
		
		return this.output.maxtir;
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

}
