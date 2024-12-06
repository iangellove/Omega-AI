package com.omega.engine.nn.layer.unet;

import java.util.List;
import java.util.Stack;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * UNet_Cond
 * @author Administrator
 *
 */
public class UNetCond extends Layer{
	
	private int[] downChannels;
	
	private int[] midChannels;
	
	private int tEmbDim;
	
	private boolean[] downSamples;
	
	private int numDowns;
	
	private int numMids;
	
	private int numUps;
	
	private boolean[] attns;
	
	private int groups = 32;
	
	private int headNum;
	
	private int convOutChannels;
	
	private int textEmbedDim;
	
	private int maxContextLen;
	
	/**
	 * layers
	 */
	public ConvolutionLayer conv_in;
	
	/**
	 * t_proj
	 */
	private FullyLayer t_linear1;
	private SiLULayer t_act;
	private FullyLayer t_linear2;
	
	/**
	 * downs
	 */
	public List<UNetDownBlockLayer> downs;
	
	/**
	 * mids
	 */
	public List<UNetMidBlockLayer> mids;
	
	/**
	 * ups
	 */
	public List<UNetUpBlockLayer> ups;
	
	/**
	 * fairs
	 */
	private GNLayer norm;
	private SiLULayer act;
	private ConvolutionLayer conv_out;
	
	private BaseKernel baseKernel;
	
	private Tensor tDiff;
	
	public UNetCond(int channel,int oChannel,int height,int width,
			int[] downChannels,int[] midChannels,boolean[] downSamples,boolean[] attns,
			int tEmbDim,int numDowns,int numMids,int numUps,int groups,int headNum,int convOutChannels,int textEmbedDim,int maxContextLen,Network network) {
		this.network = network;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.downChannels = downChannels;
		this.midChannels = midChannels;
		this.downSamples = downSamples;
		this.attns = attns;
		this.tEmbDim = tEmbDim;
		this.numDowns = numDowns;
		this.numMids = numMids;
		this.numUps = numUps;
		this.groups = groups;
		this.headNum = headNum;
		this.convOutChannels = convOutChannels;
		this.textEmbedDim = textEmbedDim;
		this.maxContextLen = maxContextLen;
		initLayers();
	}
	
	public void initLayers() {
		
		conv_in = new ConvolutionLayer(channel, downChannels[0], width, height, 3, 3, 1, 1, true, network);
		conv_in.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
		conv_in.paramsInit = ParamsInit.silu;
		
		t_linear1 = new FullyLayer(tEmbDim, tEmbDim, true, network);
		t_act = new SiLULayer(t_linear1);
		t_linear2 = new FullyLayer(tEmbDim, tEmbDim, true, network);
		
		int iw = conv_in.oWidth;
		int ih = conv_in.oHeight;
		
		Stack<Layer> downLayers = new Stack<Layer>();
		
		downLayers.push(conv_in);
		
		for(int i = 0;i<downChannels.length - 1;i++) {
			UNetDownBlockLayer down = new UNetDownBlockLayer(downChannels[i], downChannels[i+1], ih, iw, tEmbDim, downSamples[i], headNum, numDowns, groups, textEmbedDim, maxContextLen, attns[i], true, network);
			downs.add(down);
			iw = down.oWidth;
			ih = down.oHeight;
			if(i < downChannels.length - 2) {
				downLayers.push(down);
			}
		}
		
		for(int i = 0;i<midChannels.length - 1;i++) {
			UNetMidBlockLayer mid = new UNetMidBlockLayer(midChannels[i], midChannels[i+1], ih, iw, tEmbDim, headNum, numMids, groups, textEmbedDim, maxContextLen, true, network);
			mids.add(mid);
			iw = mid.oWidth;
			ih = mid.oHeight;
		}
		
		int uoc = 0;
		for(int i = midChannels.length - 1;i>=0;i--) {
			if(i != 0) {
				uoc = downChannels[i - 1];
			}else {
				uoc = convOutChannels;
			}
			UNetUpBlockLayer up = new UNetUpBlockLayer(downChannels[i] * 2, uoc, ih, iw, tEmbDim, downSamples[i], headNum, numUps, groups, textEmbedDim, maxContextLen, true, true, downLayers.pop(), network);
			ups.add(up);
			iw = up.oWidth;
			ih = up.oHeight;
		}
		
		norm = new GNLayer(groups, ups.get(ups.size() - 1));
		
		act = new SiLULayer(norm);
		
		conv_out = new ConvolutionLayer(convOutChannels, channel, width, height, 3, 3, 1, 1, true, this.network);
		conv_out.setUpdater(UpdaterFactory.create(this.network.updater, this.network.updaterParams));
		conv_out.paramsInit = ParamsInit.silu;
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
		
		this.oHeight = ih;
		this.oWidth = iw;
//		System.out.println("activeLayer.preLayer:"+activeLayer.preLayer);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(tDiff == null || tDiff.number != this.number) {
			tDiff = Tensor.createGPUTensor(tDiff, this.number, 1, 1, tEmbDim, true);
		}else {
			tDiff.clearGPU();
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

	}
	
	public void output(Tensor t,Tensor cond_input) {
		// TODO Auto-generated method stub
		/**
		 * in
		 */
		conv_in.forward(input);
		
		/**
		 * t_embd
		 */
		t_linear1.forward(t);
		t_act.forward(t_linear1.getOutput());
		t_linear2.forward(t_act.getOutput());
		
		Tensor x = conv_in.getOutput();
		
		Tensor tembd = t_linear2.getOutput();
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).forward(x, tembd, cond_input);
			x = downs.get(i).getOutput();
		}
		
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).forward(x, tembd, cond_input);
			x = mids.get(i).getOutput();
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).forward(x, tembd, cond_input);
			x = ups.get(i).getOutput();
		}
		
		/**
		 * out
		 */
		norm.forward(x);
		act.forward(norm.getOutput());
		conv_out.forward(act.getOutput());
		
		this.output = conv_out.getOutput();
	}


	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		/**
		 * out backward
		 */
		conv_out.back(delta);
		act.back(conv_out.diff);
		norm.back(act.diff);
		
		Tensor d = norm.diff;
		
		/**
		 * ups backward
		 */
		for(int i = ups.size() - 1;i>=0;i--) {
			ups.get(i).back(d, tDiff);
			d = ups.get(i).diff;
		}
		
		/**
		 * mids backward
		 */
		for(int i = mids.size() - 1;i>=0;i--) {
			mids.get(i).back(d, tDiff);
			d = mids.get(i).diff;
		}
		
		/**
		 * downs backward
		 */
		for(int i = downs.size() - 1;i>=0;i--) {
			downs.get(i).back(d, tDiff);
			d = downs.get(i).diff;
		}
		
		t_linear2.back(tDiff);
		t_act.back(t_linear2.diff);
		t_linear1.back(t_act.diff);
		
		conv_in.back(d);

		this.diff = conv_in.diff;
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
	
	public void forward(Tensor input,Tensor t,Tensor context) {
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
		this.output(t, context);
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
	public void update() {
		// TODO Auto-generated method stub
		/**
		 * in
		 */
		conv_in.update();
		
		/**
		 * t_embd
		 */
		t_linear1.update();
		t_linear2.update();
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).update();
		}
		
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).update();
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).update();
		}
		
		/**
		 * out
		 */
		norm.update();
		conv_out.update();

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
		/**
		 * in
		 */
		conv_in.accGrad(scale);
		
		/**
		 * t_embd
		 */
		t_linear1.accGrad(scale);
		t_linear2.accGrad(scale);
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).accGrad(scale);
		}
		
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).accGrad(scale);
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).accGrad(scale);
		}
		
		/**
		 * out
		 */
		norm.update();
		conv_out.accGrad(scale);
	}

}
