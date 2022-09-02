package com.omega.engine.nn.layer.active;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;

public abstract class ActiveFunctionLayer extends Layer {
	
	@Override
	public void init() {
		Layer preLayer = this.network.getPreLayer(this.index);
		if(this.parent != null) {
			preLayer = this.parent.layers.get(index - 1);
		}
		this.number = preLayer.number;
		this.channel = preLayer.oChannel;
		this.height = preLayer.oHeight;
		this.width = preLayer.oWidth;
		this.oChannel = this.channel;
		this.oHeight = this.height;
		this.oWidth = this.width;
		if(output == null || number != output.number) {
			output = new Tensor(number, oChannel, oHeight, oWidth);
		}
		if(diff == null || number != output.number) {
			diff = new Tensor(number, channel, height, width);
		}
	}
	
	@Override
	public void initBack() {
		
	}
	
}
