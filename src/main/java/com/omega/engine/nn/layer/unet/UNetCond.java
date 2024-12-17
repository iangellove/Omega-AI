package com.omega.engine.nn.layer.unet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffusion.TimeEmbeddingLayer;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;

/**
 * UNet_Cond
 * @author Administrator
 *
 */
public class UNetCond extends Layer{
	
	private int[] downChannels;
	
	private int[] midChannels;
	
	private int timeSteps;
	
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
	
	
	public TimeEmbeddingLayer t_embd;
//	/**
//	 * t_proj
//	 */
//	private FullyLayer t_linear1;
//	private SiLULayer t_act;
//	private FullyLayer t_linear2;
	
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
			int[] downChannels,int[] midChannels,boolean[] downSamples,boolean[] attns,int timeSteps,
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
		this.timeSteps = timeSteps;
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
		
		t_embd = new TimeEmbeddingLayer(timeSteps, tEmbDim, tEmbDim, true, network);
//		t_linear1 = new FullyLayer(tEmbDim, tEmbDim, true, network);
//		t_act = new SiLULayer(t_linear1);
//		t_linear2 = new FullyLayer(tEmbDim, tEmbDim, true, network);
		
		int iw = conv_in.oWidth;
		int ih = conv_in.oHeight;
		
		Stack<Layer> downLayers = new Stack<Layer>();
		
		downLayers.push(conv_in);
		
		downs = new ArrayList<UNetDownBlockLayer>();
		mids = new ArrayList<UNetMidBlockLayer>();
		ups = new ArrayList<UNetUpBlockLayer>();
		
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
		for(int i = downChannels.length - 2;i>=0;i--) {
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
		t_embd.forward(t);
//		t_linear1.forward(t);
//		t_act.forward(t_linear1.getOutput());
//		t_linear2.forward(t_act.getOutput());
		
		Tensor x = conv_in.getOutput();
//		x.showDM("conv_in");
		Tensor tembd = t_embd.getOutput();
//		tembd.showDM("tembd");
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).forward(x, tembd, cond_input);
			x = downs.get(i).getOutput();
//			x.showDM("downs:"+i);
		}
//		x.showDMByOffset(0, 100, "down");
		/**
		 * mid
		 */
		for(int i = 0;i<mids.size();i++) {
			mids.get(i).forward(x, tembd, cond_input);
			x = mids.get(i).getOutput();
		}
//		x.showDMByOffset(0, 100, "mid");
		/**
		 * ups
		 */
		for(int i = 0;i<ups.size();i++) {
			ups.get(i).forward(x, tembd, cond_input);
			x = ups.get(i).getOutput();
//			x.showDM("ups:"+i);
//			x.showDMByOffset(0, 100, "up"+i);
		}
//		x.showDMByOffset(0, 100, "up");
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
//		delta.showShape();
		conv_out.back(delta);
//		conv_out.diff.showShape();
		act.back(conv_out.diff);
//		act.diff.showDM();
		norm.back(act.diff);
		
		Tensor d = norm.diff;
//		d.showDM("norm.diff");
		/**
		 * ups backward
		 */
		for(int i = ups.size() - 1;i>=0;i--) {
			ups.get(i).back(d, tDiff);
			d = ups.get(i).diff;
		}
//		d.showDM("ups.diff");
		/**
		 * mids backward
		 */
		for(int i = mids.size() - 1;i>=0;i--) {
			mids.get(i).back(d, tDiff);
			d = mids.get(i).diff;
		}
//		d.showDM("mids.diff");
//		d.showShape();
		/**
		 * downs backward
		 */
		for(int i = downs.size() - 1;i>=0;i--) {
			downs.get(i).back(d, tDiff);
			d = downs.get(i).diff;
		}
//		d.showDM("downs.diff");
//		tDiff.showDM("tDiff");
		
		t_embd.back(tDiff);

		conv_in.back(d);
		
//		conv_in.delta.showDM("c+d");

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
		t_embd.update();
//		t_linear1.update();
//		t_linear2.update();
		
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
		t_embd.accGrad(scale);
//		t_linear1.accGrad(scale);
//		t_linear2.accGrad(scale);
		
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
	
	public static void main(String[] args) {
		
		CUDAModules.initContext();
		
		Transformer tf = new Transformer();
		tf.CUDNN = true;
		tf.RUN_MODEL = RunModel.TRAIN;
		
		int N = 2;
		int z_channels = 4;
		int H = 8;
		int W = 8;
		
		int timeSteps = 1000;
		int tEmbDim = 512;
		
		int[] downChannels = new int[] {64, 96, 128, 192};
		int[] midChannels = new int[] {192, 128};
		boolean[] downSamples = new boolean[] {true, true, true};
		boolean[] attns = new boolean[] {true, true, true};
		int numLayers = 1;
		
		int textEmbedDim = 512;
		int maxContextLen = 64;
		
		UNetCond unet = new UNetCond(z_channels, z_channels, H, W, downChannels, midChannels, downSamples, attns,
				timeSteps, tEmbDim, numLayers, numLayers, numLayers, 32, 16, 128, textEmbedDim, maxContextLen, tf);
		
		int dataLen = N * z_channels * H * W;
		
		Tensor im = new Tensor(N, z_channels, H, W, MatrixUtils.order(dataLen, 0.01f, 0.1f), true);
		
		Tensor t = new Tensor(N, 1, 1, 1, new float[] {10, 214}, true);
		
		int textLen = N * maxContextLen * textEmbedDim;
		
		Tensor context = new Tensor(N * maxContextLen, 1, 1, textEmbedDim, MatrixUtils.order(textLen, 0.01f, 0.1f), true);
		
		String weight = "H:\\model\\unet_cond.json";
		loadWeight(LagJsonReader.readJsonFileSmallWeight(weight), unet, true);
		
		tf.number = N;
		
		unet.forward(im, t, context);
		
		unet.getOutput().showDM();
		
		unet.getOutput().showShape();
		
		Tensor delta = new Tensor(N, z_channels, H, W, MatrixUtils.order(dataLen, 0.01f, 0.1f), true);
		
		unet.back(delta);
		
		
		
	}
	
	public static void loadWeight(Map<String, Object> weightMap, UNetCond unet, boolean showLayers) {
		if(showLayers) {
			for(String key:weightMap.keySet()) {
				System.out.println(key);
			}
		}
		
		/**
		 * conv_in
		 */
		ClipModelUtils.loadData(unet.conv_in.weight, weightMap, "conv_in.weight");
		ClipModelUtils.loadData(unet.conv_in.bias, weightMap, "conv_in.bias");
		
		/**
		 * t_proj
		 */
		ClipModelUtils.loadData(unet.t_embd.linear1.weight, weightMap, "t_proj.0.weight");
		ClipModelUtils.loadData(unet.t_embd.linear1.bias, weightMap, "t_proj.0.bias");
		ClipModelUtils.loadData(unet.t_embd.linear2.weight, weightMap, "t_proj.2.weight");
		ClipModelUtils.loadData(unet.t_embd.linear2.bias, weightMap, "t_proj.2.bias");
		
		/**
		 * downs
		 */
		for(int i = 0;i<3;i++) {
			unet.downs.get(i).resnetFirst.get(0).norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnetFirst.get(0).norm.gamma, weightMap, 1, "downs."+i+".resnet_conv_first.0.0.weight");
			unet.downs.get(i).resnetFirst.get(0).norm.beta = ClipModelUtils.loadData(unet.downs.get(i).resnetFirst.get(0).norm.beta, weightMap, 1, "downs."+i+".resnet_conv_first.0.0.bias");
			ClipModelUtils.loadData(unet.downs.get(i).resnetFirst.get(0).conv.weight, weightMap, "downs."+i+".resnet_conv_first.0.2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnetFirst.get(0).conv.bias, weightMap, "downs."+i+".resnet_conv_first.0.2.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).tEmbLayers.get(0).linear.weight, weightMap, "downs."+i+".t_emb_layers.0.1.weight");
			ClipModelUtils.loadData(unet.downs.get(i).tEmbLayers.get(0).linear.bias, weightMap, "downs."+i+".t_emb_layers.0.1.bias");
			
			unet.downs.get(i).resnetSecond.get(0).norm.gamma = ClipModelUtils.loadData(unet.downs.get(i).resnetSecond.get(0).norm.gamma, weightMap, 1, "downs."+i+".resnet_conv_second.0.0.weight");
			unet.downs.get(i).resnetSecond.get(0).norm.beta = ClipModelUtils.loadData(unet.downs.get(i).resnetSecond.get(0).norm.beta, weightMap, 1, "downs."+i+".resnet_conv_second.0.0.bias");
			ClipModelUtils.loadData(unet.downs.get(i).resnetSecond.get(0).conv.weight, weightMap, "downs."+i+".resnet_conv_second.0.2.weight");
			ClipModelUtils.loadData(unet.downs.get(i).resnetSecond.get(0).conv.bias, weightMap, "downs."+i+".resnet_conv_second.0.2.bias");
			
			unet.downs.get(i).attns.get(0).gn.gamma = ClipModelUtils.loadData(unet.downs.get(i).attns.get(0).gn.gamma, weightMap, 1, "downs."+i+".attention_norms.0.weight");
			unet.downs.get(i).attns.get(0).gn.beta = ClipModelUtils.loadData(unet.downs.get(i).attns.get(0).gn.beta, weightMap, 1, "downs."+i+".attention_norms.0.bias");
			
			unet.downs.get(i).attns.get(0).qLinerLayer.weight = unet.downs.get(i).attns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).attns.get(0).qLinerLayer.bias = unet.downs.get(i).attns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.downs.get(i).attns.get(0).kLinerLayer.weight = unet.downs.get(i).attns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).attns.get(0).kLinerLayer.bias = unet.downs.get(i).attns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.downs.get(i).attns.get(0).vLinerLayer.weight = unet.downs.get(i).attns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).attns.get(0).vLinerLayer.bias = unet.downs.get(i).attns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.downs.get(i).attns.get(0).oLinerLayer.weight, weightMap, "downs."+i+".attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).attns.get(0).oLinerLayer.bias, weightMap, "downs."+i+".attentions.0.out_proj.bias");
			
			unet.downs.get(i).crossAttns.get(0).gn.gamma = ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).gn.gamma, weightMap, 1, "downs."+i+".cross_attention_norms.0.weight");
			unet.downs.get(i).crossAttns.get(0).gn.beta = ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).gn.beta, weightMap, 1, "downs."+i+".cross_attention_norms.0.bias");
			
			unet.downs.get(i).crossAttns.get(0).qLinerLayer.weight = unet.downs.get(i).crossAttns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).crossAttns.get(0).qLinerLayer.bias = unet.downs.get(i).crossAttns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.downs.get(i).crossAttns.get(0).kLinerLayer.weight = unet.downs.get(i).crossAttns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).crossAttns.get(0).kLinerLayer.bias = unet.downs.get(i).crossAttns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.downs.get(i).crossAttns.get(0).vLinerLayer.weight = unet.downs.get(i).crossAttns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.downs.get(i).crossAttns.get(0).vLinerLayer.bias = unet.downs.get(i).crossAttns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).oLinerLayer.weight, weightMap, "downs."+i+".cross_attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).oLinerLayer.bias, weightMap, "downs."+i+".cross_attentions.0.out_proj.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).contextProjs.get(0).weight, weightMap, "downs."+i+".context_proj.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).contextProjs.get(0).bias, weightMap, "downs."+i+".context_proj.0.bias");
			
			unet.downs.get(i).residualInputs.get(0).weight = ClipModelUtils.loadData(unet.downs.get(i).residualInputs.get(0).weight, weightMap, 4, "downs."+i+".residual_input_conv.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).residualInputs.get(0).bias, weightMap, "downs."+i+".residual_input_conv.0.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.weight, weightMap, "downs."+i+".down_sample_conv.weight");
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.bias, weightMap, "downs."+i+".down_sample_conv.bias");
		}
		
		/**
		 * mids
		 */
		for(int i = 0;i<1;i++) {

			for(int j = 0;j<2;j++) {
				unet.mids.get(i).resnetFirst.get(j).norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnetFirst.get(j).norm.gamma, weightMap, 1, "mids."+i+".resnet_conv_first."+j+".0.weight");
				unet.mids.get(i).resnetFirst.get(j).norm.beta = ClipModelUtils.loadData(unet.mids.get(i).resnetFirst.get(j).norm.beta, weightMap, 1, "mids."+i+".resnet_conv_first."+j+".0.bias");
				ClipModelUtils.loadData(unet.mids.get(i).resnetFirst.get(j).conv.weight, weightMap, "mids."+i+".resnet_conv_first."+j+".2.weight");
				ClipModelUtils.loadData(unet.mids.get(i).resnetFirst.get(j).conv.bias, weightMap, "mids."+i+".resnet_conv_first."+j+".2.bias");
			}
			
			for(int j = 0;j<2;j++) {
				ClipModelUtils.loadData(unet.mids.get(i).tEmbLayers.get(j).linear.weight, weightMap, "mids."+i+".t_emb_layers."+j+".1.weight");
				ClipModelUtils.loadData(unet.mids.get(i).tEmbLayers.get(j).linear.bias, weightMap, "mids."+i+".t_emb_layers."+j+".1.bias");
			}
			
			for(int j = 0;j<2;j++) {
				unet.mids.get(i).resnetSecond.get(j).norm.gamma = ClipModelUtils.loadData(unet.mids.get(i).resnetSecond.get(j).norm.gamma, weightMap, 1, "mids."+i+".resnet_conv_second."+j+".0.weight");
				unet.mids.get(i).resnetSecond.get(j).norm.beta = ClipModelUtils.loadData(unet.mids.get(i).resnetSecond.get(j).norm.beta, weightMap, 1, "mids."+i+".resnet_conv_second."+j+".0.bias");
				ClipModelUtils.loadData(unet.mids.get(i).resnetSecond.get(j).conv.weight, weightMap, "mids."+i+".resnet_conv_second."+j+".2.weight");
				ClipModelUtils.loadData(unet.mids.get(i).resnetSecond.get(j).conv.bias, weightMap, "mids."+i+".resnet_conv_second."+j+".2.bias");
			}
			
			unet.mids.get(i).attns.get(0).gn.gamma = ClipModelUtils.loadData(unet.mids.get(i).attns.get(0).gn.gamma, weightMap, 1, "mids."+i+".attention_norms.0.weight");
			unet.mids.get(i).attns.get(0).gn.beta = ClipModelUtils.loadData(unet.mids.get(i).attns.get(0).gn.beta, weightMap, 1, "mids."+i+".attention_norms.0.bias");
			
			unet.mids.get(i).attns.get(0).qLinerLayer.weight = unet.mids.get(i).attns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).attns.get(0).qLinerLayer.bias = unet.mids.get(i).attns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.mids.get(i).attns.get(0).kLinerLayer.weight = unet.mids.get(i).attns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).attns.get(0).kLinerLayer.bias = unet.mids.get(i).attns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.mids.get(i).attns.get(0).vLinerLayer.weight = unet.mids.get(i).attns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).attns.get(0).vLinerLayer.bias = unet.mids.get(i).attns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.mids.get(i).attns.get(0).oLinerLayer.weight, weightMap, "mids."+i+".attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).attns.get(0).oLinerLayer.bias, weightMap, "mids."+i+".attentions.0.out_proj.bias");
			
			unet.mids.get(i).crossAttns.get(0).gn.gamma = ClipModelUtils.loadData(unet.mids.get(i).crossAttns.get(0).gn.gamma, weightMap, 1, "mids."+i+".cross_attention_norms.0.weight");
			unet.mids.get(i).crossAttns.get(0).gn.beta = ClipModelUtils.loadData(unet.mids.get(i).crossAttns.get(0).gn.beta, weightMap, 1, "mids."+i+".cross_attention_norms.0.bias");
			
			unet.mids.get(i).crossAttns.get(0).qLinerLayer.weight = unet.mids.get(i).crossAttns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).crossAttns.get(0).qLinerLayer.bias = unet.mids.get(i).crossAttns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.mids.get(i).crossAttns.get(0).kLinerLayer.weight = unet.mids.get(i).crossAttns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).crossAttns.get(0).kLinerLayer.bias = unet.mids.get(i).crossAttns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.mids.get(i).crossAttns.get(0).vLinerLayer.weight = unet.mids.get(i).crossAttns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.mids.get(i).crossAttns.get(0).vLinerLayer.bias = unet.mids.get(i).crossAttns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.mids.get(i).crossAttns.get(0).oLinerLayer.weight, weightMap, "mids."+i+".cross_attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.mids.get(i).crossAttns.get(0).oLinerLayer.bias, weightMap, "mids."+i+".cross_attentions.0.out_proj.bias");
			
			ClipModelUtils.loadData(unet.mids.get(i).contextProjs.get(0).weight, weightMap, "mids."+i+".context_proj.0.weight");
			ClipModelUtils.loadData(unet.mids.get(i).contextProjs.get(0).bias, weightMap, "mids."+i+".context_proj.0.bias");
			
			for(int j = 0;j<2;j++) {
				unet.mids.get(i).residualInputs.get(j).weight = ClipModelUtils.loadData(unet.mids.get(i).residualInputs.get(j).weight, weightMap, 4, "mids."+i+".residual_input_conv."+j+".weight");
				ClipModelUtils.loadData(unet.mids.get(i).residualInputs.get(j).bias, weightMap, "mids."+i+".residual_input_conv."+j+".bias");
			}
			
		}
		
		/**
		 * ups
		 */
		for(int i = 0;i<3;i++) {
			unet.ups.get(i).resnetFirst.get(0).norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnetFirst.get(0).norm.gamma, weightMap, 1, "ups."+i+".resnet_conv_first.0.0.weight");
			unet.ups.get(i).resnetFirst.get(0).norm.beta = ClipModelUtils.loadData(unet.ups.get(i).resnetFirst.get(0).norm.beta, weightMap, 1, "ups."+i+".resnet_conv_first.0.0.bias");
			ClipModelUtils.loadData(unet.ups.get(i).resnetFirst.get(0).conv.weight, weightMap, "ups."+i+".resnet_conv_first.0.2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnetFirst.get(0).conv.bias, weightMap, "ups."+i+".resnet_conv_first.0.2.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).tEmbLayers.get(0).linear.weight, weightMap, "ups."+i+".t_emb_layers.0.1.weight");
			ClipModelUtils.loadData(unet.ups.get(i).tEmbLayers.get(0).linear.bias, weightMap, "ups."+i+".t_emb_layers.0.1.bias");
			
			unet.ups.get(i).resnetSecond.get(0).norm.gamma = ClipModelUtils.loadData(unet.ups.get(i).resnetSecond.get(0).norm.gamma, weightMap, 1, "ups."+i+".resnet_conv_second.0.0.weight");
			unet.ups.get(i).resnetSecond.get(0).norm.beta = ClipModelUtils.loadData(unet.ups.get(i).resnetSecond.get(0).norm.beta, weightMap, 1, "ups."+i+".resnet_conv_second.0.0.bias");
			ClipModelUtils.loadData(unet.ups.get(i).resnetSecond.get(0).conv.weight, weightMap, "ups."+i+".resnet_conv_second.0.2.weight");
			ClipModelUtils.loadData(unet.ups.get(i).resnetSecond.get(0).conv.bias, weightMap, "ups."+i+".resnet_conv_second.0.2.bias");
			
			unet.ups.get(i).attns.get(0).gn.gamma = ClipModelUtils.loadData(unet.ups.get(i).attns.get(0).gn.gamma, weightMap, 1, "ups."+i+".attention_norms.0.weight");
			unet.ups.get(i).attns.get(0).gn.beta = ClipModelUtils.loadData(unet.ups.get(i).attns.get(0).gn.beta, weightMap, 1, "ups."+i+".attention_norms.0.bias");
			
			unet.ups.get(i).attns.get(0).qLinerLayer.weight = unet.ups.get(i).attns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).attns.get(0).qLinerLayer.bias = unet.ups.get(i).attns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.ups.get(i).attns.get(0).kLinerLayer.weight = unet.ups.get(i).attns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).attns.get(0).kLinerLayer.bias = unet.ups.get(i).attns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.ups.get(i).attns.get(0).vLinerLayer.weight = unet.ups.get(i).attns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).attns.get(0).vLinerLayer.bias = unet.ups.get(i).attns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.ups.get(i).attns.get(0).oLinerLayer.weight, weightMap, "ups."+i+".attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).attns.get(0).oLinerLayer.bias, weightMap, "ups."+i+".attentions.0.out_proj.bias");
			
			unet.ups.get(i).crossAttns.get(0).gn.gamma = ClipModelUtils.loadData(unet.ups.get(i).crossAttns.get(0).gn.gamma, weightMap, 1, "ups."+i+".cross_attention_norms.0.weight");
			unet.ups.get(i).crossAttns.get(0).gn.beta = ClipModelUtils.loadData(unet.ups.get(i).crossAttns.get(0).gn.beta, weightMap, 1, "ups."+i+".cross_attention_norms.0.bias");
		
			unet.ups.get(i).crossAttns.get(0).qLinerLayer.weight = unet.ups.get(i).crossAttns.get(0).qLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).crossAttns.get(0).qLinerLayer.bias = unet.ups.get(i).crossAttns.get(0).qLinerLayer.bias.createLike(1.0f);
			unet.ups.get(i).crossAttns.get(0).kLinerLayer.weight = unet.ups.get(i).crossAttns.get(0).kLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).crossAttns.get(0).kLinerLayer.bias = unet.ups.get(i).crossAttns.get(0).kLinerLayer.bias.createLike(1.0f);
			unet.ups.get(i).crossAttns.get(0).vLinerLayer.weight = unet.ups.get(i).crossAttns.get(0).vLinerLayer.weight.createLike(1.0f);
			unet.ups.get(i).crossAttns.get(0).vLinerLayer.bias = unet.ups.get(i).crossAttns.get(0).vLinerLayer.bias.createLike(1.0f);
			
			ClipModelUtils.loadData(unet.ups.get(i).crossAttns.get(0).oLinerLayer.weight, weightMap, "ups."+i+".cross_attentions.0.out_proj.weight");
			ClipModelUtils.loadData(unet.ups.get(i).crossAttns.get(0).oLinerLayer.bias, weightMap, "ups."+i+".cross_attentions.0.out_proj.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).contextProjs.get(0).weight, weightMap, "ups."+i+".context_proj.0.weight");
			ClipModelUtils.loadData(unet.ups.get(i).contextProjs.get(0).bias, weightMap, "ups."+i+".context_proj.0.bias");
			
			unet.ups.get(i).residualInputs.get(0).weight = ClipModelUtils.loadData(unet.ups.get(i).residualInputs.get(0).weight, weightMap, 4, "ups."+i+".residual_input_conv.0.weight");
			ClipModelUtils.loadData(unet.ups.get(i).residualInputs.get(0).bias, weightMap, "ups."+i+".residual_input_conv.0.bias");
			
			ClipModelUtils.loadData(unet.ups.get(i).upSampleConv.weight, weightMap, "ups."+i+".up_sample_conv.weight");
			ClipModelUtils.loadData(unet.ups.get(i).upSampleConv.bias, weightMap, "ups."+i+".up_sample_conv.bias");
		}
		
		unet.norm.gamma = ClipModelUtils.loadData(unet.norm.gamma, weightMap, 1, "norm_out.weight");
		unet.norm.beta = ClipModelUtils.loadData(unet.norm.beta, weightMap, 1, "norm_out.bias");
		ClipModelUtils.loadData(unet.conv_out.weight, weightMap, "conv_out.weight");
		ClipModelUtils.loadData(unet.conv_out.bias, weightMap, "conv_out.bias");
		
	}
	
}
