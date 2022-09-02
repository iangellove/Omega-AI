package com.omega.engine.updater;

import com.omega.common.task.ForkJobEngine;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.updater.jobs.AdamJob;

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
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(this.mw == null || this.vw == null) {
			this.mw = new float[layer.width * layer.oWidth];
			this.vw = new float[layer.width * layer.oWidth];
			if(layer.hasBias) {
				this.mb = MatrixUtils.zero(layer.oWidth);
				this.vb = MatrixUtils.zero(layer.oWidth);
			}
		}
//		
//		this.updateW(layer);
//		

		AdamJob adamJob = new AdamJob(layer.diffW.data, mw, vw, layer.weight.data, layer.learnRate, 0, layer.weight.getDataLength() - 1);
		
		ForkJobEngine.run(adamJob);

		
		if(layer.hasBias) {
			
			this.updateB(layer);
			
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
		
		AdamJob adamJob = new AdamJob(conv.diffW.data, mw, vw, conv.weight.data, conv.learnRate, 0, conv.weight.getDataLength() - 1);
		
		ForkJobEngine.run(adamJob);

		if(conv.hasBias) {

			this.updateB(conv);
			
		}

	}

	@Override
	public void updateForBN(BNLayer layer) {
		// TODO Auto-generated method stub
		
//		this.lr = layer.learnRate / layer.number;
//		System.out.println(layer.learnRate);
		/**
		 * init
		 */
		if(this.mgama == null || this.vgama == null) {
			this.mgama = MatrixUtils.zero(layer.deltaGama.length);
			this.vgama = MatrixUtils.zero(layer.deltaGama.length);
			this.mb = MatrixUtils.zero(layer.deltaBeta.length);
			this.vb = MatrixUtils.zero(layer.deltaBeta.length);
		}
		
		for(int i = 0;i<layer.deltaGama.length;i++){
			this.mgama[i] = this.beta1 * this.mgama[i] + (1 - this.beta1) * layer.deltaGama[i];
			this.vgama[i] = this.beta2 * this.vgama[i] + (1 - this.beta2) * layer.deltaGama[i] * layer.deltaGama[i];
			float mhat = this.mgama[i] / (1 - beta1);
			float vhat = this.vgama[i] / (1 - beta2);
			layer.gama[i] = layer.gama[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
		for(int i = 0;i<layer.deltaBeta.length;i++){
			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * layer.deltaBeta[i];
			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * layer.deltaBeta[i] * layer.deltaBeta[i];
			float mhat = this.mb[i] / (1 - beta1);
			float vhat = this.vb[i] / (1 - beta2);
			layer.beta[i] = layer.beta[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
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
			float mhat = this.mw[i] / (1 - beta1);
			float vhat = this.vw[i] / (1 - beta2);
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
			this.mb[i] = this.beta1 * this.mb[i] + (1 - this.beta1) * deltaB[i];
			this.vb[i] = this.beta2 * this.vb[i] + (1 - this.beta2) * deltaB[i] * deltaB[i];
			float mhat = this.mb[i] / (1 - beta1);
			float vhat = this.vb[i] / (1 - beta2);
			layer.bias.data[i] = layer.bias.data[i] - layer.learnRate * mhat / ((float)Math.sqrt(vhat) + this.eta);
		}
		
	}
	
}
