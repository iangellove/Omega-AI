package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.updater.gpu.AdamWKernel;

/**
 * Adam Updater
 * @author Administrator
 *
 */
public class AdamW extends Updater {

	private AdamWKernel kernel;
	
	private float weight_decay = 0.01f;
	
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
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate);
//		
//		System.out.print(layer.getLayerType().toString()+layer.index+":");
//		layer.weight.showDM();

		if(layer.hasBias) {
			
			kernel.updateB(layer.diffB, layer.bias, layer.network, layer.learnRate);
			
		}
		
	}

	@Override
	public void updateForMatrix(Layer layer) {
		// TODO Auto-generated method stub

	}

	@Override
	public void updateForBN(BNLayer layer) {
		// TODO Auto-generated method stub
		
//		System.out.println(layer.learnRate);
		/**
		 * init
		 */
		if(kernel == null) {
			kernel = new AdamWKernel(layer.gama.dataLength, layer.beta.dataLength, weight_decay);
		}

		kernel.updateGama(layer.diffGama, layer.gama, layer.network, layer.learnRate);
		
		kernel.updateBeta(layer.diffBeta, layer.beta, layer.network, layer.learnRate);

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
	
}
