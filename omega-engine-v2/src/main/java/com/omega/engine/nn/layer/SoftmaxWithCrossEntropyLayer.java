package com.omega.engine.nn.layer;

import com.omega.common.utils.MatrixOperation;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * SoftmaxWithCrossEntropyLayer
 * 
 * @author Administrator
 * 
 * @remark
 * 
 * The softmax function support only cross entropy loss function now.
 * 
 */
public class SoftmaxWithCrossEntropyLayer extends Layer {
	
	private double[][] currentLabel;
	
	public SoftmaxWithCrossEntropyLayer(int inputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = inputNum;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth,this.output);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = Blobs.zero(number, oChannel, oHeight, oWidth, this.diff);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		for(int n = 0;n<this.number;n++) {
			double max = MatrixOperation.max(this.input.maxtir[n][0][0]);
			double[] temp = MatrixOperation.subtraction(this.input.maxtir[n][0][0], max);
			temp = MatrixOperation.exp(temp);
			double sum = MatrixOperation.sum(temp);
			for(int i = 0;i<temp.length;i++) {
				this.output.maxtir[n][0][0][i] = temp[i] / sum;
			}
		}
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		for(int n = 0;n<this.number;n++) {
			for(int w = 0;w<this.oWidth;w++) {
				this.diff.maxtir[n][0][0][w] = this.output.maxtir[n][0][0][w] - this.currentLabel[n][w];
			}
		}
//		MatrixOperation.printImage(this.diff);
//		System.out.println(JsonUtils.toJson(this.diff));
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		
		this.output();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		this.initBack();
		this.diff();
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
	}

	public void setCurrentLabel(double[][] currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
//		System.out.println("softmax with cross layer["+this.index+"]diff start:");
////		
//		MatrixOperation.printImage(this.input);
////		
////		System.out.println("=====================================");
////		
//		MatrixOperation.printImage(this.active);
////		
//		System.out.println("=====================================");
////		
//		MatrixOperation.printImage(this.diff);
////		
//		System.out.println("softmax with cross layer["+this.index+"]diff end.");
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax_cross_entropy;
	}

}
