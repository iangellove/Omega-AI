package com.omega.engine.nn.layer.unet;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Stack;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffusion.TimeEmbeddingLayer;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.ClipVision;
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
		
		Tensor tembd = t_embd.getOutput();
		
		/**
		 * down
		 */
		for(int i = 0;i<downs.size();i++) {
			downs.get(i).forward(x, tembd, cond_input);
			x = downs.get(i).getOutput();
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
		delta.showShape();
		conv_out.back(delta);
		conv_out.diff.showShape();
		act.back(conv_out.diff);
		act.diff.showDM();
		norm.back(act.diff);
		
		Tensor d = norm.diff;
		d.showDM("norm.diff");
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
		
		t_embd.back(tDiff);
		
//		t_linear2.back(tDiff);
//		t_act.back(t_linear2.diff);
//		t_linear1.back(t_act.diff);
		
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
		
//		tf.number = N;
//		
//		unet.forward(im, t, context);
//		
//		unet.getOutput().showDM();
		
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
			
			unet.downs.get(i).crossAttns.get(0).gn.gamma = ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).gn.gamma, weightMap, 1, "downs."+i+".cross_attention_norms.0.weight");
			unet.downs.get(i).crossAttns.get(0).gn.beta = ClipModelUtils.loadData(unet.downs.get(i).crossAttns.get(0).gn.beta, weightMap, 1, "downs."+i+".cross_attention_norms.0.bias");
		
			ClipModelUtils.loadData(unet.downs.get(i).contextProjs.get(0).weight, weightMap, "downs."+i+".context_proj.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).contextProjs.get(0).bias, weightMap, "downs."+i+".context_proj.0.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).residualInputs.get(0).weight, weightMap, "downs."+i+".residual_input_conv.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).residualInputs.get(0).bias, weightMap, "downs."+i+".residual_input_conv.0.bias");
			
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.weight, weightMap, "downs."+i+".down_sample_conv.0.weight");
			ClipModelUtils.loadData(unet.downs.get(i).downSampleConv.bias, weightMap, "downs."+i+".down_sample_conv.0.bias");
		}
		
		
		
//
//		/**
//		 * embeddings
//		 */
//		loadData(network.getEncoder().getEmbeddings().getClassEmbedding(), weightMap, "embeddings.class_embedding");
//		loadData(network.getEncoder().getEmbeddings().getPatchEmbedding().weight, weightMap, "embeddings.patch_embedding.weight");
//		loadData(network.getEncoder().getEmbeddings().getPositionEmbedding().weight, weightMap, "embeddings.position_embedding.weight");
//		
//		/**
//		 * pre_layernorm
//		 */
//		network.getEncoder().getPreLayrnorm().gamma = loadData(network.getEncoder().getPreLayrnorm().gamma, weightMap, 1, "pre_layrnorm.weight");
//		network.getEncoder().getPreLayrnorm().beta = loadData(network.getEncoder().getPreLayrnorm().beta, weightMap, 1, "pre_layrnorm.bias");
//		
//		/**
//		 * encoders
//		 */
//		for(int i = 0;i<12;i++) {
//			/**
//			 * attn
//			 */
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().weight, weightMap, "encoder.layers."+i+".self_attn.q_proj.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getqLinerLayer().bias, weightMap, "encoder.layers."+i+".self_attn.q_proj.bias");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().weight, weightMap, "encoder.layers."+i+".self_attn.k_proj.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getkLinerLayer().bias, weightMap, "encoder.layers."+i+".self_attn.k_proj.bias");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().weight, weightMap, "encoder.layers."+i+".self_attn.v_proj.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getvLinerLayer().bias, weightMap, "encoder.layers."+i+".self_attn.v_proj.bias");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().weight, weightMap, "encoder.layers."+i+".self_attn.out_proj.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getAttn().getoLinerLayer().bias, weightMap, "encoder.layers."+i+".self_attn.out_proj.bias");
//			
//			/**
//			 * ln1
//			 */
//			network.getEncoder().getEncoders().get(i).getNorm1().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm1().gamma, weightMap, 1, "encoder.layers."+i+".layer_norm1.weight");
//			network.getEncoder().getEncoders().get(i).getNorm1().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm1().beta, weightMap, 1, "encoder.layers."+i+".layer_norm1.bias");
//			
//			/**
//			 * mlp
//			 */
//			loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().weight, weightMap, "encoder.layers."+i+".mlp.fc1.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear1().bias, weightMap, "encoder.layers."+i+".mlp.fc1.bias");
//			loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().weight, weightMap, "encoder.layers."+i+".mlp.fc2.weight");
//			loadData(network.getEncoder().getEncoders().get(i).getMlp().getLinear2().bias, weightMap, "encoder.layers."+i+".mlp.fc2.bias");
//			
//			/**
//			 * ln2
//			 */
//			network.getEncoder().getEncoders().get(i).getNorm2().gamma = loadData(network.getEncoder().getEncoders().get(i).getNorm2().gamma, weightMap, 1, "encoder.layers."+i+".layer_norm2.weight");
//			network.getEncoder().getEncoders().get(i).getNorm2().beta = loadData(network.getEncoder().getEncoders().get(i).getNorm2().beta, weightMap, 1, "encoder.layers."+i+".layer_norm2.bias");
////			network.getEncoder().getEncoders().get(i).getNorm2().gamma.showShape();
//		}
//		
//		/**
//		 * post_layernorm
//		 */
//		network.getEncoder().getPostLayernorm().gamma = loadData(network.getEncoder().getPostLayernorm().gamma, weightMap, 1, "post_layernorm.weight");
//		network.getEncoder().getPostLayernorm().beta = loadData(network.getEncoder().getPostLayernorm().beta, weightMap, 1, "post_layernorm.bias");
		
	}
	
}
