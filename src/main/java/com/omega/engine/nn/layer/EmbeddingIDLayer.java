package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.gpu.EmbeddingKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * 
 * FullyLayer
 * 
 * @author Administrator
 *
 */
public class EmbeddingIDLayer extends Layer{
	
	private EmbeddingKernel kernel;
	
	public EmbeddingIDLayer(int num_embeddings,int embedding_dim) {
		this.channel = 1;
		this.height = 1;
		this.width = num_embeddings;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = embedding_dim;
		this.hasParams = true;
		this.hasBias = false;
		this.initParam();
	}

	public EmbeddingIDLayer(int num_embeddings,int embedding_dim,Network network) {
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.channel = 1;
		this.height = 1;
		this.width = num_embeddings;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = embedding_dim;
		this.hasParams = true;
		this.hasBias = false;
		this.initParam();

	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.diff == null || this.number != this.diff.number){
			this.diff = new Tensor(number, channel, height, width, true, true);
		}
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number){
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(this.output == null || this.number != this.output.number){
			this.output = Tensor.createGPUTensor(this.output, number, oChannel, oHeight, oWidth, true);
		}
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		if(kernel == null) {
			kernel = new EmbeddingKernel();
		}
		this.weight = new Tensor(1, 1, width, oWidth, RandomUtils.kaiming_uniform(this.width * this.oWidth, this.width, this.paramsInit), true);
		this.diffW = new Tensor(1, 1, width, oWidth, true, true);
	}
	
	@Override
	public void output() {
		
		// TODO Auto-generated method stub
		
		if(this.input != null) {

			kernel.forward(input, weight, output);

		}
		
	}
	
	@Override
	public void diff() {
		// TODO Auto-generated method stub
		diffW.clearGPU();
		kernel.backward(delta, diffW, input);
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
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		/**
		 * 设置输入
		 */
		this.setInput(input);
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
