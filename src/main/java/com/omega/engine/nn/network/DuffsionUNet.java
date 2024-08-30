package com.omega.engine.nn.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.loss.LossFactory;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.SoftmaxWithCrossEntropyLayer;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.diffsion.ResidualBlockLayer;
import com.omega.engine.nn.layer.diffsion.TimeEmbeddingLayer;
import com.omega.engine.nn.layer.diffsion.UpSampleLayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.updater.UpdaterType;

import jcuda.Sizeof;
import jcuda.runtime.JCuda;

/**
 * Duffsion-UNet
 * @author Administrator
 *
 */
public class DuffsionUNet extends Network {
	
	private int T;
	
	private int inChannel;
	
	private int mChannel;
	
	private int[] channelMult;
	
	private int resBlockNum;
	
	private int tdim;
	
	private boolean bias = false;
	
	private Stack<Integer> chs = new Stack<Integer>();
	
	private Stack<Layer> hs = new Stack<Layer>();
	
	private InputLayer inputLayer;
	
	public TimeEmbeddingLayer temb;
	
	public ConvolutionLayer head;
	
	public List<Layer> downBlocks = new ArrayList<Layer>();
	
	public ResidualBlockLayer midResBlock1;
	public ResidualBlockLayer midResBlock2;
	
	public List<Layer> upBlocks = new ArrayList<Layer>();
	
	public GNLayer gn;
//	private BNLayer gn;
	private SiLULayer act;
	public ConvolutionLayer conv;
	
	public int width;
	
	public int height;
	
	private Tensor d_temb;
	
	public DuffsionUNet(LossType lossType,UpdaterType updater,int T,int inChannel,int mChannel,int[] channelMult,int resBlockNum,int width,int height,boolean bias) {
		this.lossFunction = LossFactory.create(lossType);
		this.bias = bias;
		this.updater = updater;
		this.T = T;
		this.width = width;
		this.height = height;
		this.inChannel = inChannel;
		this.mChannel = mChannel;
		this.channelMult = channelMult;
		this.resBlockNum = resBlockNum;
		this.tdim = mChannel * 4;
		initLayers();
	}
	
	public void initLayers() {
		
		this.inputLayer = new InputLayer(inChannel, height, width);
		
		temb = new TimeEmbeddingLayer(T, mChannel, tdim, this);
		
		head = new ConvolutionLayer(inChannel, mChannel, width, height, 3, 3, 1, 1, true, this);
		
		chs.push(mChannel);
		
		hs.push(head);
		
		/**
		 * downBlocks
		 */
		int now_ch = mChannel;
		int oHeight = height;
		int oWidth = width;
		for(int i = 0;i<channelMult.length;i++) {
			int mult = channelMult[i];
			int out_ch = mChannel * mult;
			for(int r = 0;r<resBlockNum;r++) {
				boolean attn = false;
				if(i == 2) {
					attn = true;
				}
				ResidualBlockLayer rbl = new ResidualBlockLayer(now_ch, out_ch, oHeight, oWidth, tdim, attn, this);
				downBlocks.add(rbl);
				now_ch = out_ch;
				chs.push(now_ch);
				hs.push(rbl);
				oHeight = rbl.oHeight;
				oWidth = rbl.oWidth;
			}
			now_ch = out_ch;
			if(i != channelMult.length - 1) {
				ConvolutionLayer down = new ConvolutionLayer(now_ch, now_ch, oWidth, oHeight, 3, 3, 1, 2, false, this);
				downBlocks.add(down);
				chs.push(now_ch);
				hs.push(down);
				oHeight = down.oHeight;
				oWidth = down.oWidth;
			}
		}
//		System.out.println(JsonUtils.toJson(chs));
		/**
		 * middleBlocks
		 */
		midResBlock1 = new ResidualBlockLayer(now_ch, now_ch, oHeight, oWidth, tdim, true, this);
		midResBlock2 = new ResidualBlockLayer(now_ch, now_ch, oHeight, oWidth, tdim, false, this);
		
		/**
		 * upBlocks
		 */
		int index = 0;
		for(int i = channelMult.length - 1;i>=0;i--) {
			int mult = channelMult[i];
			int out_ch = mChannel * mult;
			for(int r = 0;r<resBlockNum + 1;r++) {
				int in_ch = chs.pop();
				boolean attn = false;
				if(i == 2) {
					attn = true;
				}
				RouteLayer rl = null;
				if(index == 0) {
					rl = new RouteLayer(new Layer[] {midResBlock2, hs.pop()});
				}else {
					rl = new RouteLayer(new Layer[] {upBlocks.get(index - 1), hs.pop()});
				}
				upBlocks.add(rl);
				ResidualBlockLayer rbl = new ResidualBlockLayer(in_ch + now_ch, out_ch, oHeight, oWidth, tdim, attn, this);
				upBlocks.add(rbl);
				now_ch = out_ch;
				index = index + 2;
				oHeight = rbl.oHeight;
				oWidth = rbl.oWidth;
			}
			if(i != 0) {
				UpSampleLayer up = new UpSampleLayer(now_ch, now_ch, oHeight, oWidth, this);
				upBlocks.add(up);
				index++;
				oHeight = up.oHeight;
				oWidth = up.oWidth;
			}
		}
		
		/**
		 * tail
		 */
		gn = new GNLayer(32, this, BNType.conv_bn);
//		gn = new BNLayer(this, BNType.conv_bn);
		act = new SiLULayer(gn);
		conv = new ConvolutionLayer(now_ch, inChannel, upBlocks.get(upBlocks.size() - 1).oWidth, upBlocks.get(upBlocks.size() - 1).oHeight, 3, 3, 1, 1, false, this);
		
		this.addLayer(inputLayer);
		this.addLayer(head);
		this.addLayer(conv);
	}
	
	@Override
	public void init() throws Exception {
		// TODO Auto-generated method stub
		if(layerList.size() <= 1) {
			throw new Exception("layer size must greater than 2.");
		}
		
		this.layerCount = layerList.size();
		this.setChannel(layerList.get(0).channel);
		this.setHeight(layerList.get(0).height);
		this.setWidth(layerList.get(0).width);
	
		this.oChannel = this.getLastLayer().oChannel;
		this.oHeight = this.getLastLayer().oHeight;
		this.oWidth = this.getLastLayer().oWidth;
		
		if(layerList.get(0).getLayerType() != LayerType.input) {
			throw new Exception("first layer must be input layer.");
		}
		
		if((layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax || layerList.get(layerList.size() - 1).getLayerType() == LayerType.softmax_cross_entropy)
				&& this.lossFunction.getLossType() != LossType.cross_entropy) {
			throw new Exception("The softmax function support only cross entropy loss function now.");
		}
		
		System.out.println("the network is ready.");
	}

	@Override
	public NetworkType getNetworkType() {
		// TODO Auto-generated method stub
		return NetworkType.DUFFSION_UNET;
	}

	@Override
	public Tensor predict(Tensor input) {
		// TODO Auto-generated method stub
		this.RUN_MODEL = RunModel.TEST;
		this.forward(input);
		return this.getOutput();
	}
	
	@Override
	public Tensor forward(Tensor input) {
		return null;
	}
	
	public Tensor forward(Tensor input,Tensor t) {
//		System.out.println("en_time:"+en_time+",de_time:"+de_time);
//		input.showDMByOffset(0, 500);
//		System.err.println("input:");
//		input.showDMByOffset(0, 96);
		/**
		 * 设置输入数据
		 */
		this.setInputData(input);
		
		/**
		 * timestep embedding
		 */
		temb.forward(t);
		
		/**
		 * head
		 */
		head.forward(input);
		
//		head.getOutput().showDM();
		
//		System.out.println("head:"+MatrixOperation.isNaN(tmp.syncHost()));
		/**
		 * downsampling
		 */
		for(int i = 0;i<downBlocks.size();i++) {
			Layer layer = downBlocks.get(i);
			
			if(layer instanceof ResidualBlockLayer) {
				ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
				if(i == 0) {
					rbl.forward(head.getOutput(), temb.getOutput());
				}else {
					Layer preLayer = downBlocks.get(i - 1);
					rbl.forward(preLayer.getOutput(), temb.getOutput());
				}
			}else {
				Layer preLayer = downBlocks.get(i - 1);
				layer.forward(preLayer.getOutput());
			}
//			layer.getOutput().showDMByOffset(0, layer.getOutput().height * layer.getOutput().width);;
//			System.out.println("downsampling tmp:"+MatrixOperation.isNaN(tmp.syncHost()));
		}
		
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
//		System.out.println("downsampling:"+MatrixOperation.isNaN(tmp.syncHost()));
		
		/**
		 * middle
		 */
		midResBlock1.forward(downBlocks.get(downBlocks.size() - 1).getOutput(), temb.getOutput());
		midResBlock2.forward(midResBlock1.getOutput(), temb.getOutput());
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
//		tmp.showDMByOffset(0, 100);
//		temb.getOutput().showDMByOffset(0, 100);
		/**
		 * upsampling
		 */
//		int i = 0;
		for(int i = 0;i<upBlocks.size();i++) {
			Layer layer = upBlocks.get(i);
//			System.out.println(layer.getClass());
			if(layer instanceof ResidualBlockLayer) {
				ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
				if(i == 0) {
					rbl.forward(midResBlock2.getOutput(), temb.getOutput());
				}else {
					Layer preLayer = upBlocks.get(i - 1);
					rbl.forward(preLayer.getOutput(), temb.getOutput());
				}
//				System.err.println(i+"[tmp]:"+MatrixOperation.isNaN(tmp.syncHost()));
			}else if(layer instanceof RouteLayer) {
				RouteLayer rl = (RouteLayer) layer;
				rl.forward();
			}else {
				Layer preLayer = upBlocks.get(i - 1);
				layer.forward(preLayer.getOutput());
			}
//			System.err.println(i+":"+MatrixOperation.isNaN(tmp.syncHost()));
//			i++;
		}
//		tmp.showShape();
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
		/**
		 * tail
		 */
		gn.forward(upBlocks.get(upBlocks.size() - 1).getOutput());
		act.forward(gn.getOutput());
		conv.forward(act.getOutput());
//		System.err.println("---input---");
//		input.showDMByOffset(0, 1);
//		System.err.println("output:");
//		this.conv.getOutput().showDMByOffset(0, 1);
//		System.err.println("------");
		return this.conv.getOutput();
	}
	
	public void initBack() {
		if(d_temb == null || d_temb.number != this.number) {
			d_temb = Tensor.createGPUTensor(d_temb, this.number, temb.getOutput().channel, temb.getOutput().height, temb.getOutput().width, true);
		}
	}
	
	@Override
	public void back(Tensor lossDiff) {
		// TODO Auto-generated method stub
//		lossDiff.showDMByNumber(0);
		
		initBack();
		
		/**
		 * 设置误差
		 * 将误差值输入到最后一层
		 */
		this.setLossDiff(lossDiff);
//		System.out.println(MatrixOperation.isNaN(lossDiff.syncHost()));
//		System.err.println("lossDiff:");
//		lossDiff.showDMByOffset(0, 96*96);
		/**
		 * tail backward
		 */
		conv.back(lossDiff);
//		conv.diff.showDMByOffset(0, 96 * 96);
		act.back(conv.diff);
//		act.diff.showDMByOffset(0, 500);
		gn.back(act.diff);
//		gn.diff.showDMByOffset(0, 96 * 96);
		/**
		 * upsampling backward
		 */
//		System.err.println("back:");
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
		for(int i = upBlocks.size() - 1;i>=0;i--) {
			Layer layer = upBlocks.get(i);
			if(layer instanceof ResidualBlockLayer) {
				ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
				if(i == upBlocks.size() - 1) {
					rbl.back(gn.diff, d_temb);
				}else {
					rbl.back(upBlocks.get(i + 1).diff, d_temb);
				}
			}else {
				layer.back(upBlocks.get(i + 1).diff);
			}
		}
		
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
//		System.err.println("====upsampling tmp===");
//		tmp.showDMByOffset(0, 500);
		/**
		 * middle backward
		 */
		midResBlock2.back(upBlocks.get(0).diff, d_temb);
//		System.err.println("mh1:");
//		midResBlock2.diff.showDMByOffset(0, 100);
		midResBlock1.back(midResBlock2.diff, d_temb);
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
//		System.err.println("====middle tmp===");
//		tmp.showDMByOffset(0, 500);
//		System.err.println("mh2:");
//		midResBlock1.diff.showDMByOffset(0, 100);
		/**
		 * downsampling backward
		 */
		for(int i = downBlocks.size() - 1;i>=0;i--) {
			Layer layer = downBlocks.get(i);
			if(layer instanceof ResidualBlockLayer) {
				ResidualBlockLayer rbl = (ResidualBlockLayer) layer;
				if(i == downBlocks.size() - 1) {
					rbl.back(midResBlock1.diff, d_temb);
				}else {
					rbl.back(downBlocks.get(i + 1).diff, d_temb);
				}
			}else {
				layer.back(downBlocks.get(i + 1).diff);
			}
		}
//		System.err.println("====down tmp===");
//		tmp.showDMByOffset(0, 500);
//		System.out.println(MatrixOperation.isNaN(tmp.syncHost()));
		/**
		 * head backward
		 */
//		downBlocks.get(0).diff.showDMByOffset(0, 96 * 96);
		head.back(downBlocks.get(0).diff);
		
		/**
		 * timestep embedding backward
		 */
//		d_temb.showDM();
		temb.back(d_temb);
//		d_temb.showDMByOffset(0, 500);
		d_temb.clearGPU();

	}

	@Override
	public Tensor loss(Tensor output, Tensor label) {
		// TODO Auto-generated method stub

		switch (this.getLastLayer().getLayerType()) {
		case softmax:
//			SoftmaxLayer softmaxLayer = (SoftmaxLayer)this.getLastLayer();
//			softmaxLayer.setCurrentLabel(label);
			break;
		case softmax_cross_entropy:
			SoftmaxWithCrossEntropyLayer softmaxWithCrossEntropyLayer = (SoftmaxWithCrossEntropyLayer)this.getLastLayer();
			softmaxWithCrossEntropyLayer.setCurrentLabel(label);
			break;
		default:
			break;
		}
		
		return this.lossFunction.loss(output, label);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label) {
		// TODO Auto-generated method stub
		this.clearGrad();
		Tensor t = this.lossFunction.diff(output, label);
//		PrintUtils.printImage(t.data);
		return t;
	}
	
	public void update() {
		
		this.train_time += 1;

		temb.update();

		head.update();
		
		for(Layer layer:downBlocks) {
			layer.update();
		}

		midResBlock1.update();
		midResBlock2.update();
		
		for(Layer layer:upBlocks) {
			layer.update();
		}
		
		gn.update();
		conv.update();
		
	}

	@Override
	public void clearGrad() {
		// TODO Auto-generated method stub
		
		/**
		 * forward
		 */
		JCuda.cudaMemset(CUDAMemoryManager.workspace.getPointer(), 0, CUDAMemoryManager.workspace.getSize() * Sizeof.FLOAT);
		
		for(int i = 0;i<layerCount;i++) {
			
			Layer layer = layerList.get(i);

			if(layer.cache_delta != null) {
				layer.cache_delta.clearGPU();
			}
			
		}
		
		for(int i = 0;i<downBlocks.size();i++) {
			if(downBlocks.get(i).cache_delta != null) {
				downBlocks.get(i).cache_delta.clearGPU();
			}
		}
		
		for(int i = 0;i<upBlocks.size();i++) {
			if(upBlocks.get(i).cache_delta != null) {
				upBlocks.get(i).cache_delta.clearGPU();
			}
		}
		
		if(midResBlock1.cache_delta != null) {
			midResBlock1.cache_delta.clearGPU();
		}
		
		if(midResBlock2.cache_delta != null) {
			midResBlock2.cache_delta.clearGPU();
		}
		
		JCuda.cudaDeviceSynchronize();
	}

	@Override
	public Tensor loss(Tensor output, Tensor label, Tensor loss) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, loss);
	}

	@Override
	public Tensor lossDiff(Tensor output, Tensor label, Tensor diff) {
		// TODO Auto-generated method stub
		this.clearGrad();
		return this.lossFunction.diff(output, label, diff);
	}
	
	public Tensor loss(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.loss(output, label, igonre);
	}

	public Tensor lossDiff(Tensor output, Tensor label,int igonre) {
		// TODO Auto-generated method stub
		return this.lossFunction.diff(output, label, igonre);
	}
	
//	public void saveModel(RandomAccessFile outputStream) throws IOException {
//		decoder.saveModel(outputStream);
//		System.out.println("decoder save success...");
//		fullyLayer.saveModel(outputStream);
//		System.out.println("fullyLayer save success...");
//	}
//	
//	public void loadModel(RandomAccessFile inputStream) throws IOException {
//		decoder.loadModel(inputStream);
//		fullyLayer.loadModel(inputStream);
//	}
	
}
