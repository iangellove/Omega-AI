package com.omega.engine.nn.layer;

import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.Updater;

/**
 * Base Layer
 * @author Administrator
 *
 */
public abstract class Layer {
	
	public Network network;
	
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
	
	public Blob input;
	
	public Blob output;
	
	public Blob diff;
	
	public Blob delta;
	
	public double[][] deltaW;
	
	public double[] deltaB;
	
	public double[][] weight;
	
	public double[] bias;
	
	public double lambda = 0.01d;
	
	public double learnRate = 0.001d;
	
	public double eta = 0.00001d;
	
	public LayerType layerType;
	
	public Updater updater;
	
	public abstract void init();
	
	public abstract void initBack();
	
	public abstract void initParam();

	public abstract void output();
	
//	/**
//	 * use for gradient check
//	 * @param eta
//	 * @return
//	 */
//	public abstract Blob output(double eta);
	
	public abstract Blob getOutput();
	
	public abstract void diff();
	
	public abstract void forward();
	
	public abstract void back();
	
	public abstract void update();
	
	public abstract void showDiff();
	
	public abstract LayerType getLayerType();
	
	public void setUpdater(Updater updater) {
		this.updater = updater;
	}
	
	public void setNetwork(Network network) {
		this.network = network;
	}
	
	public void setIndex(int index) {
		this.index = index;
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setInput() {

		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.input = Blobs.transform(number, channel, height, width, this.network.getPreLayer(this.index).output);

	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta() {
		/**
		 * 获取上一层的输出作为当前层的输入
		 */
		this.delta = Blobs.transform(number, oChannel, oHeight, oWidth, this.network.getNextLayer(this.index).diff);
	}
	
	/**
	 * 转换并设置输入数据
	 */
	public void setDelta(Blob delta) {
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
	public double gradientCheck() {
//		System.out.println("*******************" + this.index + "="+this.getLayerType()+" layer********************");
//		
//		Blob output1 = this.output(eta);
//		Blob output2 = this.output(0-eta);
//		
//		double[][][][] gradientCheck = MatrixOperation.subtraction(output1.maxtir, output2.maxtir);
//		
//		gradientCheck = MatrixOperation.division(gradientCheck, 2 * this.eta);
//		
//		this.output();
//		this.diff();
//		
//		Blob currentDiff = this.diff;
//
//		System.out.println("currentDiff:"+JsonUtils.toJson(currentDiff));
//
//		System.out.println("gradientCheck:"+JsonUtils.toJson(gradientCheck));
//		
//		double finalGCError = 0.0d;
//		
//		if(this.getLayerType()!=LayerType.pooling) {
//			double[][][][] error = MatrixOperation.subtractionP(currentDiff.maxtir, gradientCheck);
//			System.out.println(JsonUtils.toJson(error));
//			finalGCError = MatrixOperation.sum(error);
//		}
//		
//		System.out.println("finalGCError:"+finalGCError);
		return 0.0d;
	}

}
