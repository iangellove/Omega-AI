package com.omega.engine.nn.layer.normalization;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;

public abstract class NormalizationLayer extends Layer {
	
	public Layer preLayer;
	
	public abstract void forward(Tensor input, int batch, int step);
	
	public abstract void back(Tensor delta, int batch, int step);
	
	public abstract void output(int batch, int step);
	
	public abstract void diff(int batch, int step);
	
	@Override
	public void init() {
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		this.number = this.network.number;
		initParam();
	}
	
	public void setPreLayer(Layer pre) {
		this.preLayer = pre;
		this.network = pre.network;
		this.channel = preLayer.oChannel;
		this.height = preLayer.oHeight;
		this.width = preLayer.oWidth;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
	}
	
	@Override
	public void initBack() {
		
	}
	
}
