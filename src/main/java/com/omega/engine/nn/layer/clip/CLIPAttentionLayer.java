package com.omega.engine.nn.layer.clip;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
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
public class CLIPAttentionLayer extends Layer{
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int dk = 0;
	
	private boolean bias = false;
	
	private FullyLayer qLinerLayer;
	private FullyLayer kLinerLayer;
	private FullyLayer vLinerLayer;
	
	private FullyLayer oLinerLayer;
	
	private DropoutLayer dropoutLayer;
	
	private DropoutLayer dropoutLayer2;
	
	private BaseKernel baseKernel;
	
	private AttentionKernel attentionKernel;
	
	private Tensor qt;
	private Tensor kt;
	private Tensor vt;
	
	private Tensor dqt;
	private Tensor dkt;
	private Tensor dvt;
	
	private Tensor temp;
	
	private Tensor attn;
	
	private Tensor oi;
	
	private Tensor dattn;
	
	private int batchSize = 1;
	
	private boolean dropout = false;
	
	public CLIPAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout) {
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
	
	public CLIPAttentionLayer(int embedDim,int headNum,int time,boolean bias,boolean dropout,Network network) {
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
		
		this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
//		this.getqLinerLayer().weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.qLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.001f, 0.001f), true);
//		Tensor qw = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.qLinerLayer.weight, qw, new int[] {0, 1, 3, 2});
//		this.qLinerLayer.weight = qw;
		
		this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
//		this.getkLinerLayer().weight = new Tensor(1, 1, nKVHeads * dk, embedDim, RandomUtils.uniform(this.embedDim * nKVHeads * dk, 0.0f, 0.02f), true);
//		this.kLinerLayer.weight = new Tensor(1, 1, nKVHeads * dk, embedDim, RandomUtils.order(this.embedDim * this.nKVHeads * dk, 0.001f, 0.001f), true);
//		Tensor kw = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.kLinerLayer.weight, kw, new int[] {0, 1, 3, 2});
//		this.kLinerLayer.weight = kw;
//		this.kLinerLayer.weight.showDM();
		
		this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
//		this.getvLinerLayer().weight = new Tensor(1, 1, nKVHeads * dk, embedDim, RandomUtils.uniform(this.embedDim * nKVHeads * dk, 0.0f, 0.02f), true);
//		this.vLinerLayer.weight = new Tensor(1, 1, nKVHeads * dk, embedDim, RandomUtils.order(this.embedDim * nKVHeads * dk, 0.001f, 0.001f), true);
//		Tensor vw = new Tensor(1, 1, embedDim, embedDim, true);
//		TensorOP.permute(this.vLinerLayer.weight, vw, new int[] {0, 1, 3, 2});
//		this.vLinerLayer.weight = vw;
//		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.vLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.01f, 2 * this.embedDim * this.embedDim * 0.01f + 0.01f), true);
//		this.vLinerLayer.weight.showDM();
		
		this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));
//		this.getoLinerLayer().weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.uniform(this.embedDim * this.embedDim, 0.0f, 0.02f), true);
//		this.oLinerLayer.weight = new Tensor(1, 1, embedDim, embedDim, RandomUtils.order(this.embedDim * this.embedDim, 0.001f, 0.001f), true);

		if(this.dropout) {
			this.dropoutLayer = new DropoutLayer(0.1f, this.network);
			this.dropoutLayer2 = new DropoutLayer(0.1f, getoLinerLayer());
		}
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}
		
		if(attentionKernel == null) {
			attentionKernel = new AttentionKernel();
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
		this.batchSize = this.number / time;
		
		if(this.qt == null || this.qt.number != this.batchSize || this.qt.height != this.time) {
			// [batch_size，time，head_num，d_k]
			this.qt = Tensor.createGPUTensor(this.qt, batchSize, headNum, time, dk, true);
			this.kt = Tensor.createGPUTensor(this.kt, batchSize, headNum, time, dk, true);
			this.vt = Tensor.createGPUTensor(this.vt, batchSize, headNum, time, dk, true);
			// [batch_size，n_heads，len_q，len_k]
			if(time < dk) {
				this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, dk, true);
			}else {
				this.temp = Tensor.createGPUTensor(this.temp, batchSize, headNum, time, time, true);
			}
			// [batch_size，n_heads，len_q，len_k]
			this.attn = Tensor.createGPUTensor(this.attn, batchSize, headNum, time, time, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.oi = Tensor.createGPUTensor(this.oi, batchSize * time, 1, 1, embedDim, true);
		}
		
		this.qt.viewOrg();
		this.kt.viewOrg();
		this.vt.viewOrg();
		if(this.getqLinerLayer().getOutput() != null) {
			this.getqLinerLayer().getOutput().viewOrg();
			this.getkLinerLayer().getOutput().viewOrg();
			this.getvLinerLayer().getOutput().viewOrg();
		}
	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.dattn == null){
//			this.dvaccum = Tensor.createGPUTensor(this.dvaccum, batchSize, headNum, time, dk, true);
			this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
			this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
			this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
			this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);
//			this.dpreatt = Tensor.createGPUTensor(this.dpreatt, batchSize, headNum, time, time, true);
//			this.dpreatt2 = Tensor.createGPUTensor(this.dpreatt2, batchSize, headNum, time, time, true);
		}else {
//			this.dvaccum.clearGPU();
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
//		System.out.println("in");
//		this.input.showDM();
		this.getqLinerLayer().forward(this.input);
		this.getkLinerLayer().forward(this.input);
		this.getvLinerLayer().forward(this.input);
		
		Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);

		TensorOP.permute(query, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(key, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(value, vt, new int[] {0, 2, 1, 3});

		scaledDotProductAttention(qt, kt, vt);
//		System.err.println("------------vaccum");
//		vaccum.showDM();
		Tensor vaccum = temp;
		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
		
		this.getoLinerLayer().forward(oi);
		
		this.output = this.getoLinerLayer().getOutput();
		
		if(dropout) {
			dropoutLayer2.forward(this.getoLinerLayer().getOutput());
			this.output = dropoutLayer2.getOutput();
		}
//		System.err.println("output:");
//		this.output.showDM();
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(dk));
		
		Tensor preatt = temp;
		
		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
		
		attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, time, d_k);

		Tensor tmp = attn;
		
		if(dropout) {
			dropoutLayer.forward(attn);
			tmp = dropoutLayer.getOutput();
		}
//		value.showDM();
		Tensor vaccum = temp;
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
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
		getqLinerLayer().update();
		getkLinerLayer().update();
		getvLinerLayer().update();
		getoLinerLayer().update();
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
		
		int embedDim = 64;
		int headNum = 8;
		int batchSize = 2;
		int time = 512;
		
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
		
		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
		
		CLIPAttentionLayer mal = new CLIPAttentionLayer(embedDim, headNum, time, false, false, tf);
		
		Tensor[] cs = RoPEKernel.getCosAndSin(time, embedDim, headNum);
		
		Tensor cos = cs[0];
		
		Tensor sin = cs[1];
		
//		mal.forward(input);
		
		for(int i = 0;i<10;i++) {
			input.showDM();
			mal.forward(input);
			
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
	
	public void saveModel(RandomAccessFile outputStream) throws IOException {
		getqLinerLayer().saveModel(outputStream);
		getkLinerLayer().saveModel(outputStream);
		getvLinerLayer().saveModel(outputStream);
		getoLinerLayer().saveModel(outputStream);
	}
	
	public void loadModel(RandomAccessFile inputStream) throws IOException {
		getqLinerLayer().loadModel(inputStream);
		getkLinerLayer().loadModel(inputStream);
		getvLinerLayer().loadModel(inputStream);
		getoLinerLayer().loadModel(inputStream);
	}

	public FullyLayer getqLinerLayer() {
		return qLinerLayer;
	}

	public void setqLinerLayer(FullyLayer qLinerLayer) {
		this.qLinerLayer = qLinerLayer;
	}

	public FullyLayer getkLinerLayer() {
		return kLinerLayer;
	}

	public void setkLinerLayer(FullyLayer kLinerLayer) {
		this.kLinerLayer = kLinerLayer;
	}

	public FullyLayer getvLinerLayer() {
		return vLinerLayer;
	}

	public void setvLinerLayer(FullyLayer vLinerLayer) {
		this.vLinerLayer = vLinerLayer;
	}

	public FullyLayer getoLinerLayer() {
		return oLinerLayer;
	}

	public void setoLinerLayer(FullyLayer oLinerLayer) {
		this.oLinerLayer = oLinerLayer;
	}

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		qLinerLayer.accGrad(scale);
		kLinerLayer.accGrad(scale);
		vLinerLayer.accGrad(scale);
		oLinerLayer.accGrad(scale);
	}
	
}
