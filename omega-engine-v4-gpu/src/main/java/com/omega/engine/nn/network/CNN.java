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

//		long start = System.nanoTime();
//		
//		long convTime = 0;
//		
//		long blTime = 0;
//		
//		long bnTime = 0;
//		
//		long fullyTime = 0;
//		
//		long poolingTime = 0;
		
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		/**
		 * forward
		 */
		for(int i = 0;i<layerCount;i++) {
			
			Layer layer = layerList.get(i);
			
//			long start2 = System.nanoTime();
			
			layer.forward();

//			System.out.println(layer.getLayerType().toString()+layer.index+":");
//			layer.output.showDM();
			
//			if(layer.getLayerType() == LayerType.conv) {
//				convTime += System.nanoTime() - start2;
//			}
//			if(layer.getLayerType() == LayerType.block) {
//				blTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.bn) {
//				bnTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.full) {
//				fullyTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.pooling) {
//				poolingTime += System.nanoTime() - start2;
//			}
//			
//			System.out.println("["+layer.getClass().toString()+"]forward:"+(System.nanoTime() - start2) / 1e6 + "ms");
			
		}
		
//		System.out.println("conv forward:"+convTime / 1e6 + "ms");
//		
//		System.out.println("block forward:"+blTime / 1e6 + "ms");
//
//		System.out.println("bn forward:"+bnTime / 1e6 + "ms");
//		
//		System.out.println("pooling forward:"+poolingTime / 1e6 + "ms");
//		
//		System.out.println("fully forward:"+fullyTime / 1e6 + "ms");
//		
//		System.out.println("all forward:"+(System.nanoTime() - start) / 1e6 + "ms");
		
		return this.getOuput();
	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		
		switch (this.getLastLayer().getLayerType()) {
		case softmax:
//			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
//			softmaxLayer.setCurrentLabel(label);
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
		
//		long start = System.nanoTime();
//		
//		long convTime = 0;
//		
//		long bnTime = 0;
//		
//		long fullyTime = 0;
//		
//		long poolingTime = 0;
//
//		long backTime = 0;
//		
//		long uct = 0;
//		
//		long bct = 0;
		
		for(int i = layerCount - 1;i>=0;i--) {
			
			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;
		
//			long start2 = System.nanoTime();
			
//			System.out.println(layer.getLayerType().toString()+layer.index+":");
			
			layer.back();

//			if(layer.getLayerType() == LayerType.conv) {
//				convTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.bn) {
//				bnTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.full) {
//				fullyTime += System.nanoTime() - start2;
//			}
//			
//			if(layer.getLayerType() == LayerType.pooling) {
//				poolingTime += System.nanoTime() - start2;
//			}
//			
//			layer.showDiff();
			
//			if(layer.weight != null) {
//				System.out.print(layer.getLayerType().toString()+layer.index+"|diffW:");
//				layer.diffW.showDM();
//			}else if(layer.getLayerType() == LayerType.bn) {
////				System.out.print(layer.getLayerType().toString()+layer.index+":");
////				BNLayer bnl = (BNLayer) layer;
////				bnl.gama.showDM();
////				bnl.beta.showDM();
//			}
			
//			System.out.println("["+layer.getClass().toString()+"]back:"+(System.nanoTime() - start) / 1e6 + "ms");
//			uct += System.nanoTime() - uStart;
//			
//			if(layer.diff!=null) {
//
//				System.out.println("["+layer.getClass().toString()+"]back:max["+MatrixOperation.max(layer.diff.maxtir)+"],min["+MatrixOperation.min(layer.diff.maxtir)+"]");
//				
//			}
//			
		}
//		
//		System.out.println("conv backward:"+convTime / 1e6 + "ms");
//		
//		System.out.println("bn backward:"+bnTime / 1e6 + "ms");
//		
//		System.out.println("pooling backward:"+poolingTime / 1e6 + "ms");
//		
//		System.out.println("fully backward:"+fullyTime / 1e6 + "ms");
//		
//		System.out.println("all backward:"+(System.nanoTime() - start) / 1e6 + "ms");
//		
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
