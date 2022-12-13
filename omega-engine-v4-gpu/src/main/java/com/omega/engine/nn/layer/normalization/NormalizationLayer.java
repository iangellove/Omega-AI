package com.omega.engine.nn.layer.normalization;

import com.omega.engine.nn.layer.Layer;

public abstract class NormalizationLayer extends Layer {
	
	public Layer preLayer;
	
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
		preLayer = pre;
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
