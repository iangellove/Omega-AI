package com.omega.engine.updater;

import com.omega.common.data.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.gpu.AdamKernel;
import com.omega.engine.updater.jobs.AdamJob;

import jcuda.driver.JCudaDriver;

/**
 * Adam Updater
 * @author Administrator
 *
 */
public class Adam extends Updater {
	
	private float beta1 = 0.9f;
	
	private float beta2 = 0.999f;
	
	private float eta = 10e-8f;
	
	private float[] mw;
	
	private float[] vw;
	
	private float[] mb;
	
	private float[] vb;
	
	private float[] mgama;
	
	private float[] vgama;
	
	private AdamJob adamJob;
	
	private AdamKernel kernel;
	
	private Tensor weight;
	
	private Tensor diffW;
	
	private Tensor bias;
	
	private Tensor diffB;
	
	public Adam(Network net) {
		this.net = net;
	}
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
//		/**
//		 * init
//		 */
//		if(this.mw == null || this.vw == null) {
//			this.mw = new float[layer.width * layer.oWidth];
//			this.vw = new float[layer.width * layer.oWidth];
//			if(layer.hasBias) {
//				this.mb = MatrixUtils.zero(layer.oWidth);
//				this.vb = MatrixUtils.zero(layer.oWidth);
//			}
//		}
////		
////		this.updateW(layer);
////		
//
//		AdamJob adamJob = new AdamJob(layer.diffW.data, mw, vw, layer.weight.data, layer.learnRate, net.number, net.train_time, 0, layer.weight.getDataLength() - 1);
//		
//		ForkJobEngine.run(adamJob);
//		
//		if(layer.hasBias) {
//			
//			this.updateB(layer);
//			
//		}
		
		if(kernel == null) {
			if(layer.hasBias) {

				kernel = new AdamKernel(layer.weight.dataLength, layer.bias.dataLength, layer.network);
				
			}else {

				kernel = new AdamKernel(layer.weight.dataLength, layer.network);
				
			}
		}
		
		if(weight == null) {
			weight = new Tensor(layer.weight.number, layer.weight.channel, layer.weight.height, layer.weight.width, layer.weight.data, true);
			diffW = new Tensor(layer.diffW.number, layer.diffW.channel, layer.diffW.height, layer.diffW.width, layer.diffW.data, true);
		}else {
			weight.setData(layer.weight.data);
			diffW.setData(layer.diffW.data);
		}
		
		kernel.updateW(diffW, weight, layer.learnRate);
		weight.syncHost();
		
//		diffW.showDM();
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			if(bias == null) {
				bias = new Tensor(1, 1, 1, layer.bias.dataLength, layer.bias.data, true);
				diffB = new Tensor(1, 1, 1, layer.diffB.dataLength, layer.diffB.data, true);
			}else {
				bias.setData(layer.bias.data);
				diffB.setData(layer.diffB.data);
			}
			kernel.updateB(diffB, bias, layer.learnRate);
			bias.syncHost();
		}
		
		
	}

	@Override
	public void updateForMatrix(Layer layer) {
		// TODO Auto-generated method stub

		if(!layer.getLayerType().equals(LayerType.conv)) {
			throw new RuntimeException("this function param must be conv layer.");
		}
		
		ConvolutionLayer conv = (ConvolutionLayer) layer;
		
		/**
		 * init
		 */
		if(this.mw == null || this.vw == null) {
			
			this.mw = new float[conv.kernelNum * conv.channel * conv.kHeight * conv.kWidth];
			this.vw = new float[conv.kernelNum * conv.channel * conv.kHeight * conv.kWidth];

			if(layer.hasBias) {
				this.mb = MatrixUtils.zero(conv.kernelNum);
				this.vb = MatrixUtils.zero(conv.kernelNum);
			}
			
		}
		
		AdamJob adamJob = new AdamJob(conv.diffW.data, mw, vw, conv.weight.data, conv.learnRate, net.number, net.train_time, 0, conv.weight.getDataLength() - 1);
		
		ForkJobEngine.run(adamJob);

		if(conv.hasBias) {

			this.updateB(conv);
			
		}

	}

	@Override
	public void updateForBN(BNLayer layer) {
		// TODO Auto-generated method stub

//		System.out.println(layer.learnRate);
//		/**
//		 * init
//		 */
//		if(this.mgama == null || this.vgama == null) {
//			this.mgama = MatrixUtils.zero(layer.deltaGama.length);
//			this.vgama = MatrixUtils.zero(layer.deltaGama.length);
//			this.mb = MatrixUtils.zero(layer.deltaBeta.length);
//			this.vb = MatrixUtils.zero(layer.deltaBeta.length);
//		}
//		
//		for(int i = 0;i<layer.deltaGama.length;i++){
//			this.mgama[i] = this.beta1 * this.mgama[i] + (1 - this.beta1) * layer.deltaGama[i];
//			this.vgama[i] = this.beta2 * this.vgama[i] + (1 - this.beta2) * layer.deltaGama[i] * layer.deltaGama[i];
//			float mhat = (float) (this.mgama[i] / (1 - Math.pow(beta1, layer.network.train_time)));
//			float vhat = (float) (this.vgama[i] / (1 - Math.pow(beta2, layer.network.train_time)));
//			layer.gama[i] = layer.gama[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
//		}
//
//		for(int i = 0;i<layer.deltaBeta.length;i++){
//			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * layer.deltaBeta[i];
//			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * layer.deltaBeta[i] * layer.deltaBeta[i];
//			float mhat = (float) (this.mb[i] / (1 - Math.pow(beta1, layer.network.train_time)));
//			float vhat = (float) (this.vb[i] / (1 - Math.pow(beta2, layer.network.train_time)));
//			layer.beta[i] = layer.beta[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
//		}
//		
		if(kernel == null) {
			kernel = new AdamKernel(layer.gama.length, layer.beta.length, layer.network);
			weight = new Tensor(1, 1, 1, layer.gama.length, layer.gama, true);
			diffW = new Tensor(1, 1, 1, layer.deltaGama.length, layer.deltaGama, true);
			bias = new Tensor(1, 1, 1, layer.beta.length, layer.beta, true);
			diffB = new Tensor(1, 1, 1, layer.deltaBeta.length, layer.deltaBeta, true);
		}else {
			weight.setData(layer.gama);
			diffW.setData(layer.deltaGama);
			bias.setData(layer.beta);
			diffB.setData(layer.deltaBeta);
		}

//		weight.showDM();

		kernel.updateGama(diffW, weight, layer.learnRate);
		kernel.updateBeta(diffB, bias, layer.learnRate);
		
		weight.syncHost();
		bias.syncHost();
		
//		System.out.println("==========deltaBeta===========");
//		diffW.showDM();
//		weight.showDM();
//		bias.showDM();
//		System.out.println(JsonUtils.toJson(layer.gama));
//		System.out.println(JsonUtils.toJson(layer.beta));
//		System.out.println("============beta=========");
		
	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.adam;
	}
	
	/**
	 * mt = beta1 * mt-1 + (1 - beta1) * gt
	 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
	 * mhat = mt / (1 - beta1)
	 * vhat = vt / (1 - beta2)
	 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
	 */
	public void updateW(Layer layer) {
		
		for(int i = 0;i<layer.diffW.getDataLength();i++) {
			this.mw[i] = this.beta1 * this.mw[i] + (1 - this.beta1) * layer.diffW.data[i];
			this.vw[i] = this.beta2 * this.vw[i] + (1 - this.beta2) * layer.diffW.data[i] * layer.diffW.data[i];
			float mhat = (float) (this.mw[i] / (1 - Math.pow(beta1, layer.network.train_time)));
			float vhat = (float) (this.vw[i] / (1 - Math.pow(beta2, layer.network.train_time)));
			layer.weight.data[i] = layer.weight.data[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}

	}
	
	/**
	 * mt = beta1 * mt-1 + (1 - beta1) * gt
	 * vt = beta2 * vt-1 + (1 - beta2) * gt^2
	 * mhat = mt / (1 - beta1)
	 * vhat = vt / (1 - beta2)
	 * W = W - learn_rate * mhat / (vhat^-1/2 + eta)
	 */
	public void updateB(Layer layer) {
		
		float[] deltaB = layer.diffB.data;

		for(int i = 0;i<deltaB.length;i++){
			float tmp = deltaB[i] / layer.network.number;
			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * tmp;
			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * tmp * tmp;
			float mhat = (float) (this.mb[i] / (1 - Math.pow(beta1, layer.network.train_time)));
			float vhat = (float) (this.vb[i] / (1 - Math.pow(beta2, layer.network.train_time)));
			layer.bias.data[i] = layer.bias.data[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
	}
	
}
