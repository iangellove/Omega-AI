package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.data.CacheDataSet;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.Updater;

/**
 * Base Layer
 * @author Administrator
 *
 */
public abstract class Layer {
	
	public Network network;
	
	public boolean PROPAGATE_DOWN = true;;
	
	public int index = 0;
	
	/**
	 * batch number
	 */
	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;
	
	public int width = 0;
	
	public int oChannel = 0;
	
	public int oHeight = 0;
	
	public int oWidth = 0;
	
	public Tensor input;
	
	public Tensor output;
	
	public Tensor diff;
	
	public Tensor delta;
	
	public Tensor weight;
	
	public Tensor bias;
	
	public Tensor diffW;
	
	public Tensor diffB;
	
	public boolean hasBias = true;
	
	public float lambda = 0.01f;
	
	public float learnRate = 0.001f;
	
	public float eta = 0.00001f;
	
	public LayerType layerType;
	
	public Updater updater;
	
	/**
	 * cache data
	 */
	private CacheDataSet tampDataSet;
	
	public boolean hasParams = false;
	
	public abstract void init();
	
	public abstract void initBack();
	
	public abstract void initParam();

	public abstract void output();
	
//	/**
//	 * use for gradient check
//	 * @param eta
//	 * @return
//	 */
//	public abstract Blob output(float eta);
	
	public abstract Tensor getOutput();
	
	public abstract void diff();
	
	public abstract void forward();
	
	public abstract void back();
	
	public abstract void forward(Tensor inpnut);
	
	public abstract void back(Tensor delta);
	
	public abstract void update();
	
	public abstract void showDiff();
	
	public abstract LayerType getLayerType();
	
	public abstract float[][][][] output(float[][][][] input);
	
	public abstract void initCache();
	
	public void setUpdater(Updater updater) {
		this.updater = updater;
	}
	
	public void setNetwork(Network network) {
		this.network = network;
	}
	
	public void setIndex(int index) {
		this.index = index;
	}
	
	public LayerInit save() {
		// TODO Auto-generated method stub
		return new LayerInit(this);
	}
	
	public void setInput(Tensor input) {
		this.input = input;
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setInput() {

		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.input = this.network.getPreLayer(this.index).output;

	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta() {
		
		if(this.delta == null) {
			/**
			 * 获取上一层的输出作为当前层的输入
			 */
			if(this.index < this.network.layerList.size() - 1) {
				this.delta = this.network.getNextLayer(this.index).diff;
			}else {
				this.delta = this.network.lossDiff;
			}

		}
		
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta(Tensor delta) {
		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.delta = delta;
	}

	/**
	 * 
	 * @Title: gradientCheck
	 *
	 * @param x
	 * @return
	 *
	 * @Description:
	 * TODO(这里用一句话描述这个方法的作用)
	 * gradientCheck:
	 * (f(x + eta) - f(x - eta)) / (2 * eta) ≈ f'(x)
	 */
	public float gradientCheck() {
		
		return 0.0f;
	}

	public CacheDataSet getTampDataSet() {
		return tampDataSet;
	}

	public void setTampDataSet(CacheDataSet tampDataSet) {
		this.tampDataSet = tampDataSet;
	}

}
