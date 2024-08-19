package com.omega.engine.nn.layer.diffsion;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.layer.normalization.BNType;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;

public class ResidualBlockLayer extends Layer{
	
	private boolean hasAttn = true;
	
	private int t_dim;
	
	private Layer[] block1;
	
	private Layer[] temb_proj;
	
	private Layer[] block2;
	
	private ConvolutionLayer shortcut;
	
	private GNLayer gn;
//	private BNLayer gn;
	
	private DuffsionAttentionBlockLayer attn;
//	private DuffsionSelfAttentionLayer2 attn;
	
	private Tensor h;
	
	private Tensor g;
	
	private Tensor dt;
	
	public ResidualBlockLayer(int channel,int oChannel,int height,int width,int t_dim,boolean hasAttn, Network network) {
		this.network = network;
		this.hasAttn = hasAttn;
		this.t_dim = t_dim;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		initLayers();
	}
	
	public void initLayers() {
		
		block1 = new Layer[3];
		block1[0] = new GNLayer(32, network, BNType.conv_bn);
//		block1[0] = new BNLayer(network, BNType.conv_bn);
		block1[1] = new SiLULayer(block1[0]);
		block1[2] = new ConvolutionLayer(channel, oChannel, width, height, 3, 3, 1, 1, false, network);
		
		temb_proj = new Layer[2];
		temb_proj[0] = new SiLULayer(network);
		temb_proj[1] = new FullyLayer(t_dim, oChannel, false, network);
		
		block2 = new Layer[3];
//		block2[0] = new BNLayer(network, BNType.conv_bn);
		block2[0] = new GNLayer(32, network, BNType.conv_bn);
		block2[1] = new SiLULayer(block2[0]);
		block2[2] = new ConvolutionLayer(oChannel, oChannel, width, height, 3, 3, 1, 1, false, network);
		
		if(channel != oChannel){
			shortcut = new ConvolutionLayer(channel, oChannel, width, height, 1, 1, 0, 1, false, network);
		}
		
		if(hasAttn) {
			gn = new GNLayer(32, network, BNType.conv_bn);
//			gn = new BNLayer(network, BNType.conv_bn);
			attn = new DuffsionAttentionBlockLayer(oChannel, width, height, false, false, network);
//			attn = new DuffsionSelfAttentionLayer(width, height , 4, oChannel, false, false, network);
//			attn = new DuffsionSelfAttentionLayer2(oChannel, 4, width * height, false, false, network);
		}

	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		initParam();
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(dt == null || dt.number != this.number) {
			dt = Tensor.createGPUTensor(dt, this.number, 1, 1, oChannel, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		if(h == null || h.number != this.number) {
			h = Tensor.createGPUTensor(h, this.number, oChannel, height, width, true);
			g = Tensor.createGPUTensor(g, this.number, oChannel, height, width, true);
		}
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
	}
	
	public void output(Tensor t) {
		// TODO Auto-generated method stub
		/**
		 * block1
		 */
//		System.out.println("input:"+MatrixOperation.isNaN(input.syncHost()));
		block1[0].forward(input);
//		System.out.println(MatrixOperation.isNaN(block1[0].getOutput().syncHost()));
		block1[1].forward(block1[0].getOutput());
//		System.out.println(MatrixOperation.isNaN(block1[1].getOutput().syncHost()));
		block1[2].forward(block1[1].getOutput());
//		System.out.println("block1[2]:"+MatrixOperation.isNaN(block1[2].getOutput().syncHost()));
		/**
		 * temb_proj
		 */
		temb_proj[0].forward(t);
		temb_proj[1].forward(temb_proj[0].getOutput());
//		System.out.println(MatrixOperation.isNaN(temb_proj[1].getOutput().syncHost()));
		
//		block1[2].getOutput().showShape();
//		temb_proj[1].getOutput().showShape();
		
		/**
		 * block1 + temb_proj
		 */
		TensorOP.add(block1[2].getOutput(), temb_proj[1].getOutput(), h, block1[2].getOutput().height * block1[2].getOutput().width);
		
//		System.out.println("h:"+MatrixOperation.isNaN(h.syncHost()));
//		h.showDMByOffset(0, 100);
//		System.err.println("------------");
//		temb_proj[1].getOutput().showShape();
//		block1[2].getOutput().showShape();
		
		/**
		 * block2
		 */
		block2[0].forward(h);
//		System.out.println("block2[0]:"+MatrixOperation.isNaN(block2[0].getOutput().syncHost()));
		block2[1].forward(block2[0].getOutput());
//		System.out.println("block2[1]:"+MatrixOperation.isNaN(block2[1].getOutput().syncHost()));
		block2[2].forward(block2[1].getOutput());
//		System.out.println("block2[2]:"+MatrixOperation.isNaN(block2[2].getOutput().syncHost()));
		/**
		 * shortcut
		 */
		Tensor tmp = input;
		if(channel != oChannel) {
//			System.out.println("===in--shutcut");
//			System.out.println("shortcut-input:"+MatrixOperation.isNaN(input.syncHost()));
////			input.showDMByOffset(0, 100);
//			System.out.println("shortcut-weight:"+MatrixOperation.isNaN(shortcut.weight.syncHost()));
//			shortcut.weight.showDMByOffset(0, 100);
			shortcut.forward(input);
			tmp = shortcut.getOutput();
//			tmp.showShape();
//			System.out.println("shortcut:"+MatrixOperation.isNaN(tmp.syncHost()));
		}
		TensorOP.add(block2[2].getOutput(), tmp, g);
		
		/**
		 * attn
		 */
		if(hasAttn) {
			gn.forward(g);
			attn.forward(gn.getOutput());
			this.output = attn.getOutput();
//			this.output.showShape();
		}else {
			this.output = g;
//			System.err.println("======");
//			System.err.println("g:"+MatrixOperation.isNaN(g.syncHost()));
//			System.err.println(MatrixOperation.isNaN(g.syncHost()));
//			g.showDMByOffset(0, 100);
//			System.err.println("======");
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
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		/**
		 * attn backward
		 */
		Tensor tmpDelta = delta;
		
		if(hasAttn) {
			attn.back(delta);
			gn.back(attn.diff);
			tmpDelta = gn.diff;
		}
			
		/**
		 * block2 backward
		 */
		block2[2].back(tmpDelta);
		block2[1].back(block2[2].diff);
		block2[0].back(block2[1].diff);
		
		/**
		 * temb_proj backward
		 */
		TensorOP.sum(block2[0].diff, dt, 2);
		temb_proj[1].back(dt);
		temb_proj[0].back(temb_proj[1].diff);
		
		/**
		 * block1 backward
		 */
		block1[2].back(block2[0].diff);
		block1[1].back(block1[2].diff);
		block1[0].back(block1[1].diff);
		
		Tensor tmp = tmpDelta;
		if(channel != oChannel) {
			shortcut.back(tmpDelta);
			tmp = shortcut.diff;
		}
		TensorOP.add(block1[0].diff, tmp, block1[0].diff);
		
		this.diff = block1[0].diff;
	}
	
	public void diff(Tensor t_diff) {
		// TODO Auto-generated method stub
//		System.out.println("index:["+index+"]("+oChannel+")"+this.delta);
		/**
		 * attn backward
		 */
		Tensor tmpDelta = delta;
		
		if(hasAttn) {
			attn.back(delta);
			gn.back(attn.diff);
			tmpDelta = gn.diff;
		}
			
		/**
		 * block2 backward
		 */
		block2[2].back(tmpDelta);
		block2[1].back(block2[2].diff);
		block2[0].back(block2[1].diff);
		
		/**
		 * temb_proj backward
		 */
		TensorOP.sum(block2[0].diff, dt, 2);
//		dt.showDMByOffset(0, 10);
		temb_proj[1].back(dt);
		temb_proj[0].back(temb_proj[1].diff);
		TensorOP.add(t_diff, temb_proj[0].diff, t_diff);
		
		/**
		 * block1 backward
		 */
		block1[2].back(block2[0].diff);
		block1[1].back(block1[2].diff);
		block1[0].back(block1[1].diff);
		
		Tensor tmp = tmpDelta;
		if(channel != oChannel) {
			shortcut.back(tmpDelta);
			tmp = shortcut.diff;
		}
		TensorOP.add(block1[0].diff, tmp, block1[0].diff);
		
		this.diff = block1[0].diff;
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
	
	public void forward(Tensor input,Tensor t) {
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
		this.output(t);
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
	
	public void back(Tensor delta,Tensor t_diff) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(t_diff);

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		block1[2].update();
		temb_proj[1].update();
		block2[2].update();
		if(channel != oChannel){
			shortcut.update();
		}
		if(hasAttn) {
			attn.update();
		}
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.duffsion_res_block;
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
	
}
