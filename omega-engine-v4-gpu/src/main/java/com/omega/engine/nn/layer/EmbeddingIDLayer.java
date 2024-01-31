package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.network.Network;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class EmbeddingIDLayer extends Layer{
	
	private BaseKernel baseKernel;
	
	public EmbeddingIDLayer(int inputNum,int outputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasParams = true;
		this.hasBias = false;
		this.initParam();
	}

	public EmbeddingIDLayer(int inputNum,int outputNum,Network network) {
		this.network = network;
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = outputNum;
		this.hasParams = true;
		this.hasBias = false;
		this.initParam();

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
		baseKernel = new BaseKernel();
		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaiming_uniform(this.width * this.oWidth, this.width, this.paramsInit), true);
		this.diffW = new Tensor(1, 1, width, oWidth, true);
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {
			
//			this.input.syncHost();
			
			for(int i=0;i<output.number;i++) {
				int wi = (int) this.input.data[i];
				baseKernel.copy_gpu(weight, output, output.getOnceSize(), wi * oWidth, 1, i * oWidth, 1);
			}

		}
		
	}
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		diffW.clearGPU();
		for(int i=0;i<input.number;i++) {
			int wi = (int) this.input.data[i];
			baseKernel.copy_gpu(delta, diffW, diffW.getOnceSize(), i * delta.getOnceSize(), 1, wi * diffW.getOnceSize(), 1);
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
		return LayerType.embedding;
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
	
	public void clear() {
//		this.output.clear();
//		this.diffW.clear();
//		this.diff.clear();
//		this.diffW.clearGPU();
	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

}