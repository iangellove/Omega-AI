package com.omega.engine.nn.layer;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.active.ActiveType;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.gpu.SoftmaxKernel;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.active.TanhLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.nn.network.Transformers;

/**
 * Multi-Head AttentionLayer
 * @author Administrator
 *
 */
public class MultiHeadAttentionLayer extends Layer{
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int qDim = 0;
	private int kDim = 0;
	private int vDim = 0;
	
	private int qProjDim = 0;
	private int kProjDim = 0;
	private int vProjDim = 0;
	private int oProjDim = 0;
	
	private int tgtLen;
	private int srcLen;
	
	private int dk = 0;
	
	private float droupout = 0.0f;
	
	private boolean bias = false;
	
	private FullyLayer qLinerLayer;
	private FullyLayer kLinerLayer;
	private FullyLayer vLinerLayer;
	
	private FullyLayer oLinerLayer;
	
	private BaseKernel baseKernel;
	
	private Tensor qt;
	private Tensor kt;
	private Tensor vt;
	
	private Tensor scores;
	
	private Tensor weights;
	
	private Tensor attn_outputs;
	
	private Tensor ot;
	
	private SoftmaxKernel softmax;
	
	private int batchSize = 1;
	
	private Tensor vt_d;
	
	
	public MultiHeadAttentionLayer(int embedDim,int headNum,int time,float droupout,boolean bias) {
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		this.bias = bias;
		this.droupout = droupout;
		this.initLayers();
	}
	
	public MultiHeadAttentionLayer(int embedDim,int headNum,int time,float droupout,boolean bias,Network network) {
		this.network = network;
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		this.bias = bias;
		this.droupout = droupout;
		this.bias = bias;
		this.initLayers();
	}
	
	public void initLayers() {
		
//		float stdv = (float) (1.0f / Math.sqrt(hiddenSize));
//		float stdv = (float) (2.0f / Math.sqrt(hiddenSize + inputSize));
		
		this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.0f), true);
//		this.qLinerLayer.bias = new Tensor(1, 1, 1, embedDim, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.1f), true);
		
		this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
//		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.02f, 0.0f), true);
//		this.kLinerLayer.weight = new Tensor(1, 1, hiddenSize, hiddenSize, RandomUtils.uniform(this.hiddenSize * this.hiddenSize, 0, stdv), true);
//		this.selfLayer.bias = new Tensor(1, 1, 1, hiddenSize, RandomUtils.uniform(this.hiddenSize, 0, stdv), true);
		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.2f), true);

		this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.01f), true);
		
		this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.val(this.embedDim * this.embedDim, 0.02f), true);
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
		
		if(softmax == null) {
			softmax = new SoftmaxKernel();
		}
		
//		System.out.println(JsonUtils.toJson(this.inputLayer.weight.syncHost()));
//		System.out.println(JsonUtils.toJson(this.inputLayer.bias.syncHost()));
//		System.out.println(JsonUtils.toJson(this.selfLayer.weight.syncHost()));
//		System.out.println(JsonUtils.toJson(this.selfLayer.bias.syncHost()));
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.dk = embedDim / headNum;
		this.batchSize = number / time;

		if(this.qt == null || this.qt.number != this.batchSize) {
			// [batch_size，time，head_num，d_k]
			this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
			this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
			this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
			// [batch_size，n_heads，len_q，len_k]
			this.scores = Tensor.createTensor(this.scores, batchSize, headNum, time, time, true);
			// [batch_size，n_heads，len_q，len_k]
			this.weights = Tensor.createTensor(this.weights, batchSize, headNum, time, time, true);
			// [batch_size, n_heads, len_q, dim_v]
			this.attn_outputs = Tensor.createTensor(this.attn_outputs, batchSize, headNum, time, dk, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.ot = Tensor.createTensor(this.ot, batchSize, time, 1, headNum * dk, true);
		}
		
	}
	
	public void init(int time,int number) {
		// TODO Auto-generated method stub
		this.number = number;
		this.time = time;
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.vt_d == null || this.vt_d.number != this.batchSize) {
			// [batch_size，time，head_num，d_k]
			this.vt_d = Tensor.createTensor(this.vt_d, batchSize, headNum, time, dk, true);
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub

//		input.showDM();

		this.qLinerLayer.forward(this.input);
		this.kLinerLayer.forward(this.input);
		this.vLinerLayer.forward(this.input);
		
		Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		
		TensorOP.permute(query, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(key, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(value, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt);
		
		TensorOP.permute(attn_outputs, ot, new int[] {0, 2, 1, 3});
		
		ot.view(batchSize * time, 1, 1, headNum * dk);
		
		this.oLinerLayer.forward(ot);
		
		this.output = this.oLinerLayer.getOutput();
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(dk));

		GPUOP.getInstance().bmm(query.getGpuData(), key.getGpuData(), scores.getGpuData(), query.number * query.channel, query.height, key.height, query.width,
				CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
		
//		query.showDM();
//		
//		scores.showDM();
		
//		scores.showShape();

		softmax.softmax(scores, weights);
		
//		weights.showShape();
//		value.showShape();
//		value.showDM();
		
		GPUOP.getInstance().bmm(weights.getGpuData(), value.getGpuData(), attn_outputs.getGpuData(), weights.number * weights.channel, weights.height, value.width, weights.width,
				CUBLAS_OP_N, CUBLAS_OP_N, 1.0f, 0.0f);
		
//		attn_outputs.showDM();
		
	}
	
	public void scaledDotProductAttentionBackward(Tensor query,Tensor key,Tensor value,Tensor delta) {
		
		// vt_diff = weightsT * delta
		GPUOP.getInstance().bmm(weights.getGpuData(), delta.getGpuData(), this.vt.getGrad().getGpuData(), weights.number * weights.channel, weights.width, delta.width, weights.height,
				CUBLAS_OP_T, CUBLAS_OP_N, 1.0f, 0.0f);

		// weights_diff = delta * vt
		GPUOP.getInstance().bmm(delta.getGpuData(), value.getGpuData(), this.weights.getGrad().getGpuData(), delta.number * delta.channel, delta.height, weights.width, delta.width,
				CUBLAS_OP_N, CUBLAS_OP_T, 1.0f, 0.0f);
		// scores_diff = softmax_backward

//		weights.view(weights.number * weights.channel, 1, 1, weights.height * weights.width);
		softmax.backward_noloss(weights, this.weights.getGrad(), scores);
		
		float d_k = (float) (1.0f / Math.sqrt(dk));
		
//		TensorOP.mul(scores, d_k, scores);

		// kt_diff = deltaT / sqrt(dk) * qt
		GPUOP.getInstance().bmm(scores.getGpuData(), query.getGpuData(), this.kt.getGrad().getGpuData(), scores.number * scores.channel, scores.width, query.width, scores.height,
				CUBLAS_OP_T, CUBLAS_OP_N, d_k, 0.0f);
		
		// qt_diff = delta / sqrt(dk) * kt
		GPUOP.getInstance().bmm(scores.getGpuData(), key.getGpuData(), this.qt.getGrad().getGpuData(), scores.number * scores.channel, scores.height, key.width, scores.width,
				CUBLAS_OP_N, CUBLAS_OP_T, d_k, 0.0f);
		
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		this.oLinerLayer.back(delta);
		
//		this.oLinerLayer.diffW.showDM();
//		
//		this.oLinerLayer.diff.showDM();
		
		this.oLinerLayer.diff.view(batchSize, time, headNum, dk);
		
		TensorOP.permute(this.oLinerLayer.diff, attn_outputs, new int[] {0, 2, 1, 3});
		
//		attn_outputs.getGrad().showDM();
		
		scaledDotProductAttentionBackward(qt, kt, vt, attn_outputs);
		
		TensorOP.permute(qt.getGrad(), this.qLinerLayer.getOutput(), new int[] {0, 2, 1, 3});
		TensorOP.permute(kt.getGrad(), this.kLinerLayer.getOutput(), new int[] {0, 2, 1, 3});
		TensorOP.permute(vt.getGrad(), this.vLinerLayer.getOutput(), new int[] {0, 2, 1, 3});
		
		Tensor queryDelta = this.qLinerLayer.getOutput().view(batchSize * time, 1, 1, headNum * dk);
		Tensor keyDelta = this.kLinerLayer.getOutput().view(batchSize * time, 1, 1, headNum * dk);
		Tensor valueDelta = this.vLinerLayer.getOutput().view(batchSize * time, 1, 1, headNum * dk);
		
		this.qLinerLayer.back(queryDelta);
		this.kLinerLayer.back(keyDelta);
		this.vLinerLayer.back(valueDelta);
		
		TensorOP.add(this.vLinerLayer.diff, this.kLinerLayer.diff, this.kLinerLayer.diff);
		TensorOP.add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
		
		this.diff = this.qLinerLayer.diff;
		
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
//		inputLayer.update(number / time);
//		selfLayer.update(number / time);
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.mutli_head_attention;
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
	
	public Tensor getWeights() {
		return weights;
	}
	
	public static void main(String[] args) {
		
		int embedDim = 512;
		int headNum = 8;
		int batchSize = 64;
		int time = 128;
		
		Transformers tf = new Transformers();
		tf.number = batchSize * time;
		
		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.0f);
		
		Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
		
		float[] delta_data = MatrixUtils.one(batchSize * time * embedDim);
		
		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
		
		MultiHeadAttentionLayer mal = new MultiHeadAttentionLayer(embedDim, headNum, time, 0.0f, false, tf);
		
		mal.forward(input);
		
//		input.showDM();
		
//		mal.getWeights().showDM();
		
		mal.getOutput().showShape();
		
//		mal.getOutput().showDM();
		
		mal.back(delta);
//		
//		mal.diff.showDM();
		
	}

}
