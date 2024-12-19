package com.omega.engine.nn.layer.vqvae;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.vae.VAEAttentionLayer;
import com.omega.engine.nn.layer.vae.VAEResnetBlock;
import com.omega.engine.nn.network.Network;

/**
 * resnet block layer
 * @author Administrator
 *
 */
public class VQVAEMidBlock extends Layer {
	
	private int numLayers;
	
	private int groups = 32;
	
	private float outputScale = 1.0f;
	
	private VAEResnetBlock res0;
	
	private List<VAEResnetBlock> resnets;
	
	private List<VAEAttentionLayer> attns;
	
	private int numHead;
	
	private boolean addAttns = false;
	
	public VQVAEMidBlock(int channel,int oChannel,int height,int width,int numLayers,int groups,int numHead,float outputScale,boolean addAttns, Network network) {
		this.network = network;
		this.addAttns = addAttns;
		this.numHead = numHead;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.groups = groups;
		this.outputScale = outputScale;
		this.numLayers = numLayers;
		
		initLayers();

	}
	
	public void initLayers() {
		
		res0 = new VAEResnetBlock(channel, oChannel, height, width, groups, outputScale, network);
		
		resnets = new ArrayList<VAEResnetBlock>(numLayers);
		
		attns = new ArrayList<VAEAttentionLayer>();
		
		int ic = oChannel;
		int ih = height;
		int iw = width;
		for(int i = 0;i<numLayers;i++) {
			
			if(addAttns) {
				VAEAttentionLayer attn = new VAEAttentionLayer(ic, numHead, ih * iw, ic, ih, iw, groups, true, false, true, network);
				attns.add(attn);
			}
			
			VAEResnetBlock res = new VAEResnetBlock(ic, oChannel, ih, iw, groups, 1.0f, network);
			resnets.add(res);
			ic = oChannel;
			ih = res.oHeight;
			iw = res.oWidth;
		}
		
		this.oHeight = ih;
		this.oWidth = iw;
		
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

		res0.forward(this.input);
		
		Tensor x = res0.getOutput();

		for(int i = 0;i<numLayers;i++) {
			if(addAttns) {
				attns.get(i).forward(x);
				x = attns.get(i).getOutput();
			}
			resnets.get(i).forward(x);
			x = resnets.get(i).getOutput();
		}
		
		this.output = x;
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
		Tensor dx = delta;
		for(int i = numLayers - 1;i>=0;i--) {
			resnets.get(i).back(dx);
			dx = resnets.get(i).diff;
			if(addAttns) {
				attns.get(i).back(dx);
				dx = attns.get(i).diff;
			}
		}

		res0.back(dx);
		
		this.diff = res0.diff;
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
		
		res0.update();
		
		for(int i = 0;i<numLayers;i++) {
			if(addAttns) {
				attns.get(i).update();
			}
			resnets.get(i).update();
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