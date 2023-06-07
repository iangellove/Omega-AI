package com.omega.engine.nn.network;

import com.omega.common.data.Tensor;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.updater.UpdaterType;

/**
 * Convolutional Neural Networks
 * @author Administrator
 *
 */
public class CNN extends Network {
	
	public CNN(LossFunction lossFunction) {
		this.lossFunction = lossFunction;
	}
	
	public CNN(LossFunction lossFunction,UpdaterType updater) {
		this.lossFunction = lossFunction;
		this.updater = updater;
	}
	
	public CNN(LossType lossType,UpdaterType updater) {
		this.lossFunction = LossFactory.create(lossType);
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
		Layer outputLayer = layerList.get(layerList.size() - 1);
		this.channel = inputLayer.channel;
		this.height = inputLayer.height;
		this.width = inputLayer.width;
		this.oChannel = outputLayer.oChannel;
		this.oHeight = outputLayer.oHeight;
		this.oWidth = outputLayer.oWidth;
		
		System.out.println("the network is ready.");
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
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		
		switch (this.getLastLayer().getLayerType()) {
		case softmax:
			break;
		case softmax_cross_entropy:
			SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer)this.getLastLayer();
			softmaxWithCrossEntropyLayer.setCurrentLabel(label);
			break;
		default:
			break;
		}
		
		return this.lossFunction.loss(output, label);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label);
	}

	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub

		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);

		for(int i = layerCount - 1;i>=0;i--) {
			
			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;

			layer.back();

		}

	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOuput();
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.CNN;
	}
	
}
