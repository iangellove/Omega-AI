package com.omega.engine.nn.layer.diffsion;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.EmbeddingIDLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.SiLULayer;
import com.omega.engine.nn.network.Network;

/**
 * diffsion model TimeEmbeddingLayer
 * @author Administrator
 *
 */
public class TimeEmbeddingLayer extends Layer{
	
	private int T;
	
	private int d_model;
	
	private int dim;
	
	private EmbeddingIDLayer emb;
	private FullyLayer linear1;
	private SiLULayer act;
	private FullyLayer linear2;
	
	public TimeEmbeddingLayer(int T,int d_model,int dim, Network network) {
		this.network = network;
		this.T = T;
		this.d_model = d_model;
		this.dim = dim;
		this.height = 1;
		this.width = T;
		this.oHeight = 1;
		this.oWidth = dim;
		initLayers();
	}

	public void initLayers() {

		emb = new EmbeddingIDLayer(T, d_model, true, network);
		emb.weight = emb.createTimeEMB(T, d_model);
		
		linear1 = new FullyLayer(d_model, dim, false, network);
		
		act = new SiLULayer(linear1);
		
		linear2 = new FullyLayer(dim, dim, false, network);

	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
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
		emb.forward(input);
		linear1.forward(emb.getOutput());
		act.forward(linear1.getOutput());
		linear2.forward(act.getOutput());
		this.output = linear2.getOutput();
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
		linear2.back(delta);
		act.back(linear2.diff);
		linear1.back(act.diff);
//		this.diff = linear1.diff;
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
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		
		/**
		 * 设置输入
		 */
		this.setInput(inpnut);

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
	public void update() {
		// TODO Auto-generated method stub
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
		return LayerType.time_embedding;
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
