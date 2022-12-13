package com.omega.engine.updater;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.updater.gpu.SGDKernel;

public class SGDM extends Updater {

	private SGDKernel kernel;
	
	@Override
	public void update(Layer layer) {
		// TODO Auto-generated method stub
		/**
		 * init
		 */
		if(kernel == null) {
			
			if(layer.hasBias) {

				kernel = new SGDKernel(layer.weight.dataLength, layer.bias.dataLength);
				
			}else {

				kernel = new SGDKernel(layer.weight.dataLength);
				
			}
			
		}
		
		kernel.updateW(layer.diffW, layer.weight, layer.network, layer.learnRate);

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
		/**
		 * init
		 */
		if(kernel == null) {
			kernel = new SGDKernel(layer.gama.dataLength, layer.beta.dataLength);
			kernel.weight_decay = 0.0f;
		}

		kernel.updateW(layer.diffGama, layer.gama, layer.network, layer.learnRate);
		
		kernel.updateB(layer.diffBeta, layer.beta, layer.network, layer.learnRate);

	}

	@Override
	public UpdaterType getUpdaterType() {
		// TODO Auto-generated method stub
		return UpdaterType.sgd;
	}

}
