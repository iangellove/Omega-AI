package com.omega.engine.nn.layer.unet;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.ConvolutionTransposeLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * UNetUpBlockLayer
 * @author Administrator
 *
 */
public class UNetUpBlockLayer extends Layer{
	
	private boolean attn = false;
	
	private boolean crossAttn = false;
	
	private int tEmbDim;
	
	private int contextDim = 0;
	
	private int maxContextLen = 0;
	
	private boolean upSample;
	
	private int numHeads;
	
	private int numLayers = 1;
	
	private int groups = 32;
	
	private RouteLayer cat;
	
	private List<UNetResnetBlockLayer> resnetFirst;
	
	private List<UNetTEmbLayer> tEmbLayers;
	
	private List<UNetResnetBlockLayer> resnetSecond;
	
	private List<UNetAttentionLayer> attns;
	
	private List<UNetAttentionLayer> crossAttns;
	
	private List<FullyLayer> contextProjs;
	
	private List<ConvolutionLayer> residualInputs;
	
	private ConvolutionTransposeLayer upSampleConv;
	
	private BaseKernel baseKernel;
	
	private Tensor[] t_out;
	
	private Tensor[] res_out;
	
	private Tensor kvDiff;
	
	public UNetUpBlockLayer(int channel,int oChannel,int height,int width,int tEmbDim,boolean upSample,int numHeads,int numLayers,int groups,int contextDim,int maxContextLen,boolean attn,boolean crossAttn,Layer catLayer,Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		this.tEmbDim = tEmbDim;
		this.contextDim = contextDim;
		this.maxContextLen = maxContextLen;
		this.upSample = upSample;
		this.numHeads = numHeads;
		this.numLayers = numLayers;
		this.groups = groups;
		this.attn = attn;
		this.crossAttn = crossAttn;
		initLayers(catLayer);
	}
	
	public void initLayers(Layer catLayer) {
		
		Layer[] catLayers = null;
		
		if(upSample) {
			upSampleConv = new ConvolutionTransposeLayer(channel/2, channel/2, width, height, 4, 4, 1, 2, 1, 0, true, network);
			upSampleConv.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
			upSampleConv.paramsInit = ParamsInit.silu;

			this.oHeight = upSampleConv.oHeight;
			this.oWidth = upSampleConv.oWidth;
			
			catLayers = new Layer[] {upSampleConv, catLayer};
		}else {
			catLayers = new Layer[] {this, catLayer};
		}
		
		cat = new RouteLayer(catLayers);
		
		resnetFirst = new ArrayList<UNetResnetBlockLayer>();
		resnetSecond = new ArrayList<UNetResnetBlockLayer>();
		
		for(int i = 0;i<numLayers;i++) {
			int ic = oChannel;
			if(i == 0) {
				ic = channel;
			}
			UNetResnetBlockLayer b = new UNetResnetBlockLayer(ic, oChannel, height, width, groups, network);
			resnetFirst.add(b);
		}
		
		if(tEmbDim > 0) {
			tEmbLayers = new ArrayList<UNetTEmbLayer>();
			for(int i = 0;i<numLayers;i++) {
				UNetTEmbLayer t = new UNetTEmbLayer(tEmbDim, oChannel, network);
				tEmbLayers.add(t);
			}
		}
		
		for(int i = 0;i<numLayers;i++) {
			UNetResnetBlockLayer b = new UNetResnetBlockLayer(oChannel, oChannel, height, width, groups, network);
			resnetSecond.add(b);
		}
		
		if(attn) {
			attns = new ArrayList<UNetAttentionLayer>();
			for(int i = 0;i<numLayers;i++) {
				UNetAttentionLayer a = new UNetAttentionLayer(oChannel, numHeads, oChannel, height, width, groups, true, false, true, network);
				attns.add(a);
			}
		}
		
		if(crossAttn) {
			contextProjs = new ArrayList<FullyLayer>();
			for(int i = 0;i<numLayers;i++) {
				FullyLayer f = new FullyLayer(contextDim, oChannel, true, network);
				contextProjs.add(f);
			}
			crossAttns = new ArrayList<UNetAttentionLayer>();
			for(int i = 0;i<numLayers;i++) {
				UNetAttentionLayer a = new UNetAttentionLayer(oChannel, oChannel, oChannel, numHeads, maxContextLen, oChannel, height, width, groups, true, false, true, network);
				crossAttns.add(a);
			}
		}
		
		residualInputs = new ArrayList<ConvolutionLayer>();
		for(int i = 0;i<numLayers;i++) {
			int ic = oChannel;
			if(i == 0) {
				ic = channel;
			}
			ConvolutionLayer c = new ConvolutionLayer(ic, oChannel, width, height, 1, 1, 0, 1, true, network);
			c.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
			c.paramsInit = ParamsInit.silu;
			residualInputs.add(c);
		}

		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}

	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		if(res_out == null || res_out[0].number != this.number) {
			if(res_out == null) {
				res_out = new Tensor[numLayers];
			}
			for(int i = 0;i<numLayers;i++) {
				res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
			}
		}
		
		if(tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
			if(t_out == null) {
				t_out = new Tensor[numLayers];
			}
			for(int i = 0;i<numLayers;i++) {
				t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
			}
		}
		
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		if(res_out == null || res_out[0].number != this.number) {
			if(res_out == null) {
				res_out = new Tensor[numLayers];
			}
			for(int i = 0;i<numLayers;i++) {
				res_out[i] = Tensor.createGPUTensor(res_out[i], this.number, oChannel, height, width, true);
			}
		}
		
		if(tEmbDim > 0 && (t_out == null || t_out[0].number != this.number)) {
			if(t_out == null) {
				t_out = new Tensor[numLayers];
			}
			for(int i = 0;i<numLayers;i++) {
				t_out[i] = Tensor.createGPUTensor(t_out[i], this.number, oChannel, height, width, true);
			}
		}
		
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(crossAttn && (kvDiff == null || kvDiff.number * maxContextLen != this.number * maxContextLen)) {
			kvDiff = Tensor.createGPUTensor(kvDiff, this.number * maxContextLen, 1, 1, oChannel, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		Tensor x = input;
		
		if(upSample) {
			upSampleConv.forward(x);
			x = upSampleConv.getOutput();
		}
		
		cat.forward(x);
		
		x = cat.getOutput();

		for(int i = 0;i<numLayers;i++) {
			Tensor res_i = x;
			resnetFirst.get(i).forward(x);
			resnetSecond.get(i).forward(resnetFirst.get(i).getOutput());
			residualInputs.get(i).forward(res_i);
			
			TensorOP.add(resnetSecond.get(i).getOutput(), residualInputs.get(i).getOutput(), res_out[i]);
			
			x = res_out[i];
			
			if(attn) {
				attns.get(i).forward(x);
				x = attns.get(i).getOutput();
			}
			
		}
		
		this.output = x;
	}
	
	public void output(Tensor tembd,Tensor context) {
		// TODO Auto-generated method stub
		Tensor x = input;
		
		if(upSample) {
			upSampleConv.forward(x);
			x = upSampleConv.getOutput();
		}
		
		cat.forward(x);
		
		x = cat.getOutput();
		
		for(int i = 0;i<numLayers;i++) {
			
			Tensor res_i = x;
			
			resnetFirst.get(i).forward(x);
			
			tEmbLayers.get(i).forward(tembd);
			
			Tensor r1 = resnetFirst.get(i).getOutput();
			
			TensorOP.add(r1, tEmbLayers.get(i).getOutput(), t_out[i], r1.height * r1.width);
			
			resnetSecond.get(i).forward(t_out[i]);
			
			residualInputs.get(i).forward(res_i);
			
			TensorOP.add(resnetSecond.get(i).getOutput(), residualInputs.get(i).getOutput(), res_out[i]);
			
			x = res_out[i];
			
			if(attn) {
				attns.get(i).forward(x);
				x = attns.get(i).getOutput();
			}
			
			if(crossAttn) {
				contextProjs.get(i).forward(context);
				Tensor cp = contextProjs.get(i).getOutput();
				crossAttns.get(i).forward(x, cp, cp);
				x = crossAttns.get(i).getOutput();
			}
			
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
	}
	
	public void diff(Tensor tDiff) {
		// TODO Auto-generated method stub
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		
		Tensor d = delta;
		
		for(int i = numLayers - 1;i>=0;i--) {
			
			if(crossAttn) {
				crossAttns.get(i).back(d, kvDiff);
				contextProjs.get(i).back(kvDiff);
				d = crossAttns.get(i).diff;
			}
			
			if(attn) {
				attns.get(i).back(d);
				d = attns.get(i).diff;
			}
			
			residualInputs.get(i).back(d);
			
			resnetSecond.get(i).back(d);
			
			d = resnetSecond.get(i).diff;
			
			if(tEmbDim > 0) {
				tEmbLayers.get(i).back(d);
				TensorOP.add(tDiff, tEmbLayers.get(i).diff, tDiff);
			}
			
			resnetFirst.get(i).back(d);
			
			d = resnetFirst.get(i).diff;
			
			TensorOP.add(d, residualInputs.get(i).diff, d);
			
		}
		
		cat.back(d);
		
		d = cat.diff;
		
		if(upSample) {
			upSampleConv.back(d);
			d = upSampleConv.diff;
		}
		
		this.diff = d;

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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output();
	}
	
	public void forward(Tensor input,Tensor tembd,Tensor context) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output(tembd, context);
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
	
	public void back(Tensor delta,Tensor tDiff) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(tDiff);

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		
		upSampleConv.update();
		
		for(int i = 0;i<numLayers;i++) {
			
			resnetFirst.get(i).update();
			
			if(tEmbDim > 0) {
				tEmbLayers.get(i).update();
			}
			
			
			resnetSecond.get(i).update();
			
			residualInputs.get(i).update();
			
			if(attn) {
				attns.get(i).update();
			}
			
			if(crossAttn) {
				contextProjs.get(i).update();
				crossAttns.get(i).update();
			}
			
		}
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.unet_down;
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
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		
		upSampleConv.accGrad(scale);
		
		for(int i = 0;i<numLayers;i++) {
			
			resnetFirst.get(i).accGrad(scale);
			
			if(tEmbDim > 0) {
				tEmbLayers.get(i).accGrad(scale);
			}
			
			
			resnetSecond.get(i).accGrad(scale);
			
			residualInputs.get(i).accGrad(scale);
			
			if(attn) {
				attns.get(i).accGrad(scale);
			}
			
			if(crossAttn) {
				contextProjs.get(i).accGrad(scale);
				crossAttns.get(i).accGrad(scale);
			}
			
		}

	}

}
