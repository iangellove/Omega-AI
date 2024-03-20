package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.active.GeluLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.LNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.updater.UpdaterFactory;

/**
 * Transformer Decoder Layer
 * @author Administrator
 *
 */
public class TransformerBlock extends Layer{
	
	private int time;
	
	private int headNum = 8;
	
	private int embedDim = 0;
	
	private boolean bias = false;
	
	private CausalSelfAttentionLayer attn;
	private LNLayer ln1;
	
	/**
	 * mlp
	 */
	private FullyLayer linear1;
	private GeluLayer gelu1;
//	private LeakyReluLayer gelu1;
	private FullyLayer linear2;
	private LNLayer ln2;
	
	private BaseKernel baseKernel;
	
	private Tensor tmp1;
	
	private Tensor tmp2;
	
	public TransformerBlock(int headNum,int time,int embedDim,boolean bias) {
		this.headNum = headNum;
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public TransformerBlock(int headNum,int time,int embedDim,boolean bias,Network network) {
		this.headNum = headNum;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.attn = new CausalSelfAttentionLayer(embedDim, headNum, time, bias, network);

		this.ln1 = new LNLayer(attn);
		
		this.linear1 = new FullyLayer(embedDim, 4 * embedDim, bias, network);
		this.linear1.weight = new Tensor(1, 1, embedDim, 4 * embedDim, RandomUtils.uniform(this.embedDim * 4 * this.embedDim, 0.0f, 0.02f), true);
		
		this.gelu1 = new GeluLayer(this.linear1);
		
//		this.gelu1 = new LeakyReluLayer(this.linear1);
		
		this.linear2 = new FullyLayer(4 * embedDim, embedDim, bias, network);
		this.linear2.weight = new Tensor(1, 1, 4 * embedDim, embedDim, RandomUtils.uniform(this.embedDim * 4 * this.embedDim, 0.0f, 0.02f), true);
		
		
		this.ln2 = new LNLayer(linear2);
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}

	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.input.number;
		if(this.tmp1 == null || this.tmp1.number != this.number) {
//			System.out.println(number);
			this.tmp1 = Tensor.createTensor(this.tmp1, number, 1, 1, embedDim, true);
			this.tmp2 = Tensor.createTensor(this.tmp2, number, 1, 1, embedDim, true);
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		attn.forward(input);
		
		TensorOP.add(attn.getOutput(), input, tmp1);
		
		ln1.forward(tmp1);
		
		linear1.forward(ln1.getOutput());
		
		gelu1.forward(linear1.getOutput());
		
		linear2.forward(gelu1.getOutput());
		
		TensorOP.add(linear2.getOutput(), ln1.getOutput(), tmp2);
		
		ln2.forward(tmp2);
		
		this.output = ln2.getOutput();
		
	}
	
	public void output(Tensor mask) {
		// TODO Auto-generated method stub

		attn.forward(input, mask);
		
		TensorOP.add(attn.getOutput(), input, tmp1);
		
		ln1.forward(tmp1);
		
		linear1.forward(ln1.getOutput());
		
		gelu1.forward(linear1.getOutput());
		
		linear2.forward(gelu1.getOutput());
		
		TensorOP.add(linear2.getOutput(), ln1.getOutput(), tmp2);
		
		ln2.forward(tmp2);
		
		this.output = ln2.getOutput();
	}
	
	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		ln2.back(delta);
		
		linear2.back(ln2.diff);
		
		gelu1.back(linear2.diff);
		
		linear1.back(gelu1.diff);
		
		TensorOP.add(linear1.diff, ln2.diff, tmp2);
		
		ln1.back(tmp2);
		
		attn.back(ln1.diff);
		
		TensorOP.add(attn.diff, ln1.diff, tmp2);
		
		this.diff = tmp2;
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 计算输出
		 */
		this.output();
	}
	
	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 计算输出
		 */
		this.output();
		
	}
	
	public void forward(Tensor input,Tensor mask) {
		// TODO Auto-generated method stub
		/**
		 * 设置输入
		 */
		this.setInput(input);
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 计算输出
		 */
		this.output(mask);
		
	}
	
	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		ln1.update();
		attn.update();
		ln2.update();
		linear1.update();
		linear2.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.transformer_decoder;
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
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}
	
}
