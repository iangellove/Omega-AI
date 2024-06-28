package com.omega.engine.nn.layer;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * CausalSelfAttentionLayer
 * @author Administrator
 *
 */
public class LlamaCausalSelfAttentionLayer extends Layer{
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int dk = 0;
	
	private boolean bias = false;
	
	private FullyLayer qLinerLayer;
	private FullyLayer kLinerLayer;
	private FullyLayer vLinerLayer;
//	private FullyLayer qkvLinerLayer;
	
	private FullyLayer oLinerLayer;
	
	private DropoutLayer dropoutLayer;
	
	private DropoutLayer dropoutLayer2;
	
	private BaseKernel baseKernel;
	
	private AttentionKernel attentionKernel;
	
	private RoPEKernel ropeKernel;
	
	private Tensor rq;
	private Tensor rk;
	
	private Tensor qt;
	private Tensor kt;
	private Tensor vt;
	
	private Tensor dqt;
	private Tensor dkt;
	private Tensor dvt;
	
	private Tensor vaccum;
	
	private Tensor preatt;
	
	private Tensor attn;
	
	private Tensor oi;
	
	private Tensor dvaccum;
	
	private Tensor dattn;
	
	private Tensor dpreatt;

	private int batchSize = 1;
	
	private boolean dropout = false;
	
	public LlamaCausalSelfAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout) {
		this.bias = bias;
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		if(embedDim % headNum != 0){
			throw new RuntimeException("embedDim % headNum must be zero.");
		}
		this.dk = embedDim / headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.dropout = dropout;
		this.initLayers();
	}
	
	public LlamaCausalSelfAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout,Network network) {
		this.bias = bias;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.time = time;
		this.embedDim = embedDim;
		this.headNum = headNum;
		if(embedDim % headNum != 0){
			throw new RuntimeException("embedDim % headNum must be zero.");
		}
		this.dk = embedDim / headNum;
		this.bias = bias;
		this.oChannel = 1;
		this.oHeight = 1;
		this.oWidth = embedDim;
		this.dropout = dropout;
		this.initLayers();
	}
	
	public void initLayers() {
		
		this.qLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);
//		this.qLinerLayer.weight.showDM();
		
		this.kLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.kLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, this.embedDim * this.embedDim * 0.01f + 0.01f), true);
//		this.kLinerLayer.weight.showDM();
		
		this.vLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 2 * this.embedDim * this.embedDim * 0.01f + 0.01f), true);
//		this.vLinerLayer.weight.showDM();
		
		this.oLinerLayer = new FullyLayer(embedDim, embedDim, bias, this.network);
		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 0.01f), true);

		if(this.dropout) {
			this.dropoutLayer = new DropoutLayer(0.1f, this.network);
			this.dropoutLayer2 = new DropoutLayer(0.1f, oLinerLayer);
		}
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
		
		if(attentionKernel == null) {
			attentionKernel = new AttentionKernel();
		}
		
		if(ropeKernel == null) {
			ropeKernel = new RoPEKernel();
		}
		
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub

	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		this.time = this.network.time;
		this.batchSize = number / time;
		
		if(this.preatt == null || this.preatt.number != this.batchSize || this.preatt.width != this.time) {
			// [batch_size，time，head_num，d_k]
			this.rq = Tensor.createGPUTensor(this.rq, batchSize, time, headNum, dk, true);
			this.rk = Tensor.createGPUTensor(this.rk, batchSize, time, headNum, dk, true);
			this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
			this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
			this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
			// [batch_size，n_heads，len_q，len_k]
			this.preatt = Tensor.createGPUTensor(this.preatt, batchSize, headNum, time, time, true);
			// [batch_size，n_heads，len_q，len_k]
			this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
			// [batch_size, n_heads, len_q, dim_v]
			this.vaccum = Tensor.createGPUTensor(this.vaccum, batchSize, headNum, time, dk, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
		}
		
		this.qt.viewOrg();
		this.kt.viewOrg();
		this.vt.viewOrg();
		this.rq.viewOrg();
		this.rk.viewOrg();
		if(this.qLinerLayer.getOutput() != null) {
			this.qLinerLayer.getOutput().viewOrg();
			this.kLinerLayer.getOutput().viewOrg();
			this.vLinerLayer.getOutput().viewOrg();
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.dvaccum == null){
			this.dvaccum = Tensor.createGPUTensor(this.dvaccum, batchSize, headNum, time, dk, true);
			this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
			this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
			this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
			this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
			this.dpreatt = Tensor.createGPUTensor(this.dpreatt, batchSize, headNum, time, time, true);
		}else {
			this.dvaccum.clearGPU();
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
	
	public void output(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
//		System.out.println("in");
//		this.input.showDM();
		this.qLinerLayer.forward(this.input);
		this.kLinerLayer.forward(this.input);
		this.vLinerLayer.forward(this.input);
		
		Tensor query = this.qLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.kLinerLayer.getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.vLinerLayer.getOutput().view(batchSize, time, headNum, dk);

		/**
		 * apply RoPE
		 */
		ropeKernel.forward(cos, sin, query, key, rq, rk);
		
		TensorOP.permute(rq, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(rk, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(value, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt);
		
		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
		
		this.oLinerLayer.forward(oi);
		
		this.output = this.oLinerLayer.getOutput();
		
		if(dropout) {
			dropoutLayer2.forward(this.oLinerLayer.getOutput());
			this.output = dropoutLayer2.getOutput();
		}
		
//		this.output.showDMByNumber(0);
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(dk));
		
		GPUOP.getInstance().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);

		attentionKernel.scale(preatt, d_k, batchSize, headNum, time);

		attentionKernel.softmax_forward(preatt, attn, batchSize, headNum, time);
		
		Tensor tmp = attn;
		
		if(dropout) {
			dropoutLayer.forward(attn);
			tmp = dropoutLayer.getOutput();
		}

		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
	}

	public void scaledDotProductAttentionBackward() {
		
		Tensor tmp = attn;
		
		if(dropout) {
			tmp = dropoutLayer.getOutput();
		}
		
	    // backward into datt
		GPUOP.getInstance().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);

		// backward into dv
		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);

		if(dropout) {
			dropoutLayer.back(dattn);
			dattn = dropoutLayer.diff;
		}
		
		// backward into preatt
		attentionKernel.softmax_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum);
		
		// backward into q
		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
		
		// backward into k
		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub

	}
	
	public void diff(Tensor cos,Tensor sin) {
		// TODO Auto-generated method stub
		
		if(dropout) {
			dropoutLayer2.back(delta);
			this.oLinerLayer.back(dropoutLayer2.diff, oi);
		}else {
			this.oLinerLayer.back(delta, oi);
		}

		attentionKernel.unpermute_backward(dvaccum, oi, batchSize, time, headNum, dk);
		
		scaledDotProductAttentionBackward();

		qt.view(this.qLinerLayer.getOutput().shape());
		kt.view(this.kLinerLayer.getOutput().shape());
		vt.view(this.vLinerLayer.getOutput().shape());
		
		TensorOP.permute(dqt, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(dkt, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(dvt, vt, new int[] {0, 2, 1, 3});
		
		/**
		 * RoPE backward
		 */
		ropeKernel.backward(cos, sin, qt, kt, rq, rk);
		
		Tensor queryDelta = rq.view(batchSize * time, 1, 1, headNum * dk);
		Tensor keyDelta = rk.view(batchSize * time, 1, 1, headNum * dk);
		Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
		
		this.qLinerLayer.back(queryDelta);
		this.kLinerLayer.back(keyDelta);
		this.vLinerLayer.back(valueDelta);
		
		TensorOP.add(this.qLinerLayer.diff, this.kLinerLayer.diff, this.qLinerLayer.diff);
		TensorOP.add(this.qLinerLayer.diff, this.vLinerLayer.diff, this.qLinerLayer.diff);

		this.diff = this.qLinerLayer.diff;
		
//		this.diff.showDMByNumber(0);
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub

	}
	
	@Override
	public void back() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub

	}
	
	public void forward(Tensor cos,Tensor sin,Tensor input) {
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
		this.output(cos, sin);
		
	}
	
	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

	}
	
	public void back(Tensor cos,Tensor sin,Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(cos, sin);
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		qLinerLayer.update();
		kLinerLayer.update();
		vLinerLayer.update();
		oLinerLayer.update();
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
	
//	public Tensor getWeights() {
//		return weights;
//	}

	public static void main(String[] args) {
		
		int embedDim = 4;
		int headNum = 2;
		int batchSize = 3;
		int time = 3;
		
		Transformer tf = new Transformer();
		tf.number = batchSize * time;
		tf.time = time;
		
		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
		
//		int[] rts = new int[] {2, 3, 3};
//		
//		for(int b = 0;b<batchSize;b++) {
//			int rt = rts[b];
//			for(int t = 0;t<time;t++) {
//				if(t > rt) {
//					for(int n = 0;n<embedDim;n++) {
//						data[b * time * embedDim + t * embedDim + n] = 0;
//					}
//				}
//			}
//		}
		
//		float[] maskData = new float[] {1,1,1,0,0,1,1,1,1,0,1,1,1,1,0};
//		
//		Tensor mask = new Tensor(batchSize, 1, time, time, maskData, true);
		
		Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
		
//		input.showDM();
		
		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
		
		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
		
		LlamaCausalSelfAttentionLayer mal = new LlamaCausalSelfAttentionLayer(embedDim, headNum, time, false, false, tf);
		
		Tensor[] cs = RoPEKernel.getCosAndSin(time, embedDim, headNum);
		
		Tensor cos = cs[0];
		
		Tensor sin = cs[1];
		
//		mal.forward(input);
		
		for(int i = 0;i<10;i++) {

			mal.forward(cos, sin, input);
			
//			input.showDM();
			
//			mal.getWeights().showDM();
			
			mal.getOutput().showShape();
			
			mal.getOutput().showDM();
			
			mal.back(cos, sin, delta);
//			delta.showDM();
			mal.diff.showDM();
//			delta.copyData(tmp);
		}
		
	}
	
	public static boolean same(Tensor a,Tensor b) {
		float[] ad = a.syncHost();
		float[] bd = b.syncHost();
		for(int i=0;i<ad.length;i++) {
			if(ad[i] != bd[i]) {
				System.out.println(ad[i]+":"+bd[i] + "["+i+"]");
				return false;
			}
		}
		return true;
	}

}
