package com.omega.engine.updater;

import java.util.Map;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.NormalizationLayer;
import com.omega.engine.updater.gpu.AdamWKernel;

/**
 * Adam Updater
 * @author Administrator
 *
 */
public class AdamW extends Updater {

	private AdamWKernel kernel;
	
	private float weight_decay = 0.0005f;
	
	public AdamW(Map<String,Float> params) {
		this.params = params;
	}
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(kernel == null) {
			
			if(layer.hasBias) {

				kernel = new AdamWKernel(layer.weight.dataLength, layer.bias.dataLength, weight_decay);
				
			}else {

				kernel = new AdamWKernel(layer.weight.dataLength, weight_decay);
				
			}
			
			kernel.setParams(params);
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate);
//		layer.diffW.clearGPU();
//		
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			
			kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate);
//			layer.diffB.clearGPU();
		}
		
	}

	@Override
	public void updateForMatrix(Layer layer) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateForBN(NormalizationLayer layer) {
		// TODO Auto-generated method stub
		
//		System.out.println(layer.learnRate);
		/**
		 * init
		 */
		if(kernel == null) {
			kernel = new AdamWKernel(layer.gamma.dataLength, layer.beta.dataLength, weight_decay);
		}
		
		kernel.setParams(params);

		kernel.updateGama(layer.diffGamma, layer.gamma, layer.network, layer.learnRate);
		
		kernel.updateBeta(layer.diffBeta, layer.beta, layer.network, layer.learnRate);

		layer.diffGamma.clearGPU();
		layer.diffBeta.clearGPU();
		
//		
//		System.out.println("==========diffBeta===========");
//		diffW.showDM();
//		weight.showDM();
//		layer.diffBeta.showDM();
//		layer.beta.showDM();
//		System.out.println("============beta=========");
//		
	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.adamw;
	}

	@Override
	public void update(Layer layer, int batchSize) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(kernel == null) {
			
			if(layer.hasBias) {

				kernel = new AdamWKernel(layer.weight.dataLength, layer.bias.dataLength, weight_decay);
				
			}else {

				kernel = new AdamWKernel(layer.weight.dataLength, weight_decay);
				
			}
			
			kernel.setParams(params);
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate, batchSize);
//		layer.diffW.clearGPU();
//		
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			
			kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate, batchSize);
//			layer.diffB.clearGPU();
		}
		
	}
	
}
