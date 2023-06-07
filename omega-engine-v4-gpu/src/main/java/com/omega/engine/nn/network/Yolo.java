package com.omega.engine.nn.network;

import com.omega.common.data.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.updater.UpdaterType;

/**
 * yolo model
 * @author Administrator
 *
 */
public class Yolo extends OutputsNetwork{
	
	private LossFunction[] losses; 
	
	private LossType lossType;
	
	private Tensor[] loss;
	
	private Tensor[] lossDiff;
	
	public Yolo(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}
	
	public Yolo(LossType lossType,UpdaterType updater) {
		this.lossType = lossType;
		this.updater = updater;
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub

		if(layerList.size() <= 0) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		
		if(layerList.get(0).getLayerType() != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}
		
		Layer inputLayer = layerList.get(0);

		this.channel = inputLayer.channel;
		this.height = inputLayer.height;
		this.width = inputLayer.width;
		
		this.losses = LossFactory.create(lossType, outputLayers);
		
		if(this.loss == null) {
			this.loss = new Tensor[this.outputNum];
		}
		
		if(this.lossDiff == null){
			this.lossDiff = new Tensor[this.outputNum];
		}
		
		System.out.println("the network is ready.");
	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		/**
		 * forward
		 */
		for(int i = 0;i<layerCount;i++) {
			
			Layer layer = layerList.get(i);

			layer.forward();

		}

		return this.getOuput();
	}
	
	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void back(Tensor[] lossDiffs) {

		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiffs);

		for(int i = layerCount - 1;i>=0;i--) {
			
			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;

			layer.back();

		}

	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Tensor[] loss(Tensor label) {
		// TODO Auto-generated method stub
		for(int i = 0;i<losses.length;i++) {
			this.loss[i] = losses[i].loss(getOutputs()[i], label);
		}
		return this.loss;
	}
	
	public Tensor[] loss(Tensor[] outputs,Tensor label) {
		// TODO Auto-generated method stub
		for(int i = 0;i<losses.length;i++) {
			this.loss[i] = losses[i].loss(outputs[i], label);
		}
		return this.loss;
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}
	
	@Override
	public Tensor[] lossDiff(Tensor label) {
		// TODO Auto-generated method stub
		for(int i = 0;i<losses.length;i++) {
			this.lossDiff[i] = losses[i].diff(getOutputs()[i], label);
		}
		return this.lossDiff;
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.YOLO;
	}
	
	@Override
	public Tensor[] predicts(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		/**
		 * forward
		 */
		for(int i = 0;i<layerCount;i++) {
			
			Layer layer = layerList.get(i);

			layer.forward();

		}

		return getOutputs();
	}

}
