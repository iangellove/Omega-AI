package com.omega.engine.loss;

import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;

/**
 * Cross Entropy loss function
 * 
 * @author Administrator
 *
 * @loss: - ∑ y * ln(f(x))
 * @diff: - ∑ y * (1 / f(x))
 *
 */
public class CrossEntropyLoss extends LossFunction {

	public final LossType lossType = LossType.cross_entropy;
	
	private static CrossEntropyLoss instance;
	
	public static CrossEntropyLoss operation() {
		if(instance == null) {
			instance = new CrossEntropyLoss();
		}
		return instance;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.cross_entropy;
	}

	@Override
	public Blob loss(Blob x, double[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		
		for(int n = 0;n<x.number;n++) {
			for(int w = 0;w<x.width;w++) {
				temp.maxtir[n][0][0][w] = - label[n][w] * Math.log(x.maxtir[n][0][0][w]);
			}
		}

		return temp;
	}

	@Override
	public Blob diff(Blob x, double[][] label) {
		// TODO Auto-generated method stub
		Blob temp = Blobs.blob(x.number,x.channel,x.height,x.width);
		for(int n = 0;n<x.number;n++) {
			for(int w = 0;w<x.width;w++) {
				temp.maxtir[n][0][0][w] = - label[n][w] / x.maxtir[n][0][0][w];
			}
		}
		return temp;
	}
	
	public static void main(String[] args) {
		double[][] x = new double[][] {{0.2,0.7,0.1},{0.8,0.1,0.1},{0.5,0.4,0.1},{0.3,0.3,0.4}};
		Blob xb = Blobs.blob(4, 1, 1, 3, x);
		double[][] label = new double[][] {{0,1,0},{1,0,0},{1,0,0},{0,0,1}};
		double error = CrossEntropyLoss.operation().gradientCheck(xb,label);
		System.out.println("error:"+error);
	}

}
