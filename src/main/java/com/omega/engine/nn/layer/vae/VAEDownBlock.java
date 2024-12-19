package com.omega.engine.nn.layer.vae;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class VAEDownBlock extends Layer {
	
	private int numLayers;
	
	private int group = 32;
	
	private List<VAEResnetBlock> resnets;
	
	private boolean addDownsampler = false;
	
	private ConvolutionLayer downsampler;
	
	private float outputScale = 1.0f;
	
	public VAEDownBlock(int channel,int oChannel,int height,int width,int numLayers,int group,float outputScale,boolean addDownsampler, Network network) {
		this.network = network;
		this.addDownsampler = addDownsampler;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.group = group;
		this.outputScale = outputScale;
		this.numLayers = numLayers;
		
		initLayers();
		
	}
	
	public void initLayers() {
		
		resnets = new ArrayList<VAEResnetBlock>(numLayers);
		
		int ic = channel;
		int ih = height;
		int iw = width;
		for(int i = 0;i<numLayers;i++) {
			VAEResnetBlock res = new VAEResnetBlock(ic, oChannel, ih, iw, group, outputScale, network);
			resnets.add(res);
			ic = oChannel;
			ih = res.oHeight;
			iw = res.oWidth;
		}
		
		this.oHeight = ih;
		this.oWidth = iw;
		
		if(addDownsampler) {
			downsampler = new ConvolutionLayer(ic, oChannel, iw, ih, 3, 3, 1, 2, false, this.network); 
			downsampler.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
			downsampler.paramsInit = ParamsInit.silu;
			this.oHeight = downsampler.oHeight;
			this.oWidth = downsampler.oWidth;
		}

	}

	@Override
	public void init() {
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub

	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		Tensor x = this.input;
		
		for(int i = 0;i<numLayers;i++) {
			resnets.get(i).forward(x);
			x = resnets.get(i).getOutput();
		}
		
		this.output = x;
		
		if(addDownsampler) {
			downsampler.forward(x);
			this.output = downsampler.getOutput();
		}
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		System.out.println(index);
		Tensor diffOut = delta;
		if(addDownsampler) {
			downsampler.back(diffOut);
			diffOut = downsampler.diff;
		}
		
		for(int i = numLayers - 1;i>=0;i--) {
			resnets.get(i).back(diffOut);
			diffOut = resnets.get(i).diff;
		}
		
		this.diff = diffOut;
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		
		/**
		 * 参数初始化
		 */
		this.init();
		
		/**
		 * 设置输入
		 */
		this.setInput();

		/**
		 * 计算输出
		 */
		this.output();
		
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		for(int i = 0;i<numLayers;i++) {
			resnets.get(i).update();
		}
		
		if(addDownsampler) {
			downsampler.update();
		}
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.block;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub

	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub

		/**
		 * 参数初始化
		 */
		this.init();
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output();
		
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();

	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
	}

}