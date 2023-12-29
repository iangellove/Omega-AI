package com.omega.engine.nn.layer.active;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;

public abstract class ActiveFunctionLayer extends Layer {
	
	public Layer preLayer;
	
	public abstract void forward(Tensor input,int batch,int step);
	
	public abstract void back(Tensor delta,int batch,int step);
	
	public abstract void output(int batch,int step);
	
	public abstract void diff(int batch,int step);
	
	@Override
	public void init() {

		this.number = this.network.number;
		
		if(this.preLayer == null) {
			this.preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}

		if(output == null || number != output.number) {
			this.output = Tensor.createTensor(this.output, number, oChannel, oHeight, oWidth, true);
//			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}

	}
	
	@Override
	public void initBack() {
//		System.out.println(this.index);
		if(this.diff == null) {
			this.diff = this.network.getDelta(this.index);
		}
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
	public void update() {
		// TODO Auto-generated method stub
		
	}
	
}
