//package com.omega.engine.nn.layer.diffsion;
//
//import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
//import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
//
//import com.omega.common.data.Tensor;
//import com.omega.engine.ad.op.TensorOP;
//import com.omega.engine.gpu.BaseKernel;
//import com.omega.engine.gpu.GPUOP;
//import com.omega.engine.nn.layer.ConvolutionLayer;
//import com.omega.engine.nn.layer.DropoutLayer;
//import com.omega.engine.nn.layer.Layer;
//import com.omega.engine.nn.layer.LayerType;
//import com.omega.engine.nn.layer.gpu.AttentionKernel;
//import com.omega.engine.nn.network.Network;
//import com.omega.engine.updater.UpdaterFactory;
//
///**
// * CausalSelfAttentionLayer
// * @author Administrator
// *
// */
//public class DuffsionSelfAttentionLayer extends Layer{
//	
//	private int time;
//	
//	private int width = 0;
//	
//	private int height = 0;
//	
//	private boolean bias = false;
//	
//	private ConvolutionLayer qLayer;
//	private ConvolutionLayer kLayer;
//	private ConvolutionLayer vLayer;
//	private ConvolutionLayer oLayer;
//	
////	private FullyLayer qkvLinerLayer;
////	
////	private FullyLayer oLinerLayer;
//	
//	private DropoutLayer dropoutLayer;
//	
//	private DropoutLayer dropoutLayer2;
//	
//	private BaseKernel baseKernel;
//	
//	private AttentionKernel attentionKernel;
//	
//	private Tensor qt;
//	private Tensor kt;
//	private Tensor vt;
//	
//	private Tensor dqt;
//	private Tensor dkt;
//	private Tensor dvt;
//	
//	private Tensor vaccum;
//	
//	private Tensor preatt;
//	
//	private Tensor attn;
//	
//	private Tensor oi;
//	
//	private Tensor dvaccum;
//	
//	private Tensor dattn;
//	
//	private Tensor dpreatt;
//	
//	private int batchSize = 1;
//	
//	private boolean dropout = false;
//	
//	public DuffsionSelfAttentionLayer(int width,int height,int headNum,int time,boolean bias,boolean dropout) {
//		this.bias = bias;
//		this.time = time;
//		this.channel = time;
//		this.height = height;
//		this.width = width;
//		this.bias = bias;
//		this.oChannel = 1;
//		this.oHeight = height;
//		this.oWidth = width;
//		this.dropout = dropout;
//		this.initLayers();
//	}
//	
//	public DuffsionSelfAttentionLayer(int width,int height,int headNum,int time,boolean bias,boolean dropout,Network network) {
//		this.bias = bias;
//		this.network = network;
//		if(this.updater == null) {
//			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
//		}
//		this.time = time;
//		this.channel = time;
//		this.height = height;
//		this.width = width;
//		this.bias = bias;
//		this.oChannel = 1;
//		this.oHeight = height;
//		this.oWidth = width;
//		this.dropout = dropout;
//		this.initLayers();
//	}
//	
//	public void initLayers() {
//		this.qLayer = new ConvolutionLayer(time, time, width, height, 1, 1, 0, 1, bias, this.network);
//		this.kLayer = new ConvolutionLayer(time, time, width, height, 1, 1, 0, 1, bias, this.network);
//		this.vLayer = new ConvolutionLayer(time, time, width, height, 1, 1, 0, 1, bias, this.network);
//		
//		this.oLayer = new ConvolutionLayer(time, time, width, height, 1, 1, 0, 1, bias, this.network);
//		
//		if(this.dropout) {
//			this.dropoutLayer = new DropoutLayer(0.1f, this.network);
//			this.dropoutLayer2 = new DropoutLayer(0.1f, oLayer);
//		}
//		
//		if(baseKernel == null) {
//			baseKernel = new BaseKernel();
//		}
//		
//		if(attentionKernel == null) {
//			attentionKernel = new AttentionKernel();
//		}
//		
//	}
//	
//	@Override
//	public void init() {
//		// TODO Auto-generated method stub
//		this.number = this.network.number;
//		this.batchSize = number;
//
//		if(this.preatt == null || this.preatt.number != this.batchSize || this.preatt.width != this.time) {
//			// [batch_size，time，head_num，d_k]
//			this.qt = Tensor.createTensor(this.qt, batchSize, height, width, time, true);
//			this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
//			this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
//			// [batch_size，n_heads，len_q，len_k]
//			this.preatt = Tensor.createTensor(this.preatt, batchSize, headNum, time, time, true);
//			// [batch_size，n_heads，len_q，len_k]
//			this.attn = Tensor.createTensor(this.attn, batchSize, headNum, time, time, true);
//			// [batch_size, n_heads, len_q, dim_v]
//			this.vaccum = Tensor.createTensor(this.vaccum, batchSize, headNum, time, dk, true);
//			// [batch_size, len_q, n_heads * dim_v]
//			this.oi = Tensor.createTensor(this.oi, batchSize, time, headNum, dk, true);
//		}
//
//	}
//	
//	public void init(Tensor input) {
//		// TODO Auto-generated method stub
//		this.number = input.number;
//		this.batchSize = number;
//		
//		if(this.preatt == null || this.preatt.number != this.batchSize || this.preatt.width != this.time) {
//			// [batch_size，time，head_num，d_k]
//			this.qt = Tensor.createTensor(this.qt, batchSize, headNum, time, dk, true);
//			this.kt = Tensor.createTensor(this.kt, batchSize, headNum, time, dk, true);
//			this.vt = Tensor.createTensor(this.vt, batchSize, headNum, time, dk, true);
//			// [batch_size，n_heads，len_q，len_k]
//			this.preatt = Tensor.createTensor(this.preatt, batchSize, headNum, time, time, true);
//			// [batch_size，n_heads，len_q，len_k]
//			this.attn = Tensor.createTensor(this.attn, batchSize, headNum, time, time, true);
//			// [batch_size, n_heads, len_q, dim_v]
//			this.vaccum = Tensor.createTensor(this.vaccum, batchSize, headNum, time, dk, true);
//			// [batch_size, len_q, n_heads * dim_v]
//			this.oi = Tensor.createTensor(this.oi, batchSize, time, height, width, true);
//		}
//
//	}
//	
//	@Override
//	public void initBack() {
//		// TODO Auto-generated method stub
//		if(this.dvaccum == null){
//			this.dvaccum = Tensor.createTensor(this.dvaccum, batchSize, headNum, time, dk, true);
//			this.dqt = Tensor.createTensor(this.dqt, batchSize, headNum, time, dk, true);
//			this.dkt = Tensor.createTensor(this.dkt, batchSize, headNum, time, dk, true);
//			this.dvt = Tensor.createTensor(this.dvt, batchSize, headNum, time, dk, true);
//			this.dattn = Tensor.createTensor(this.dattn, batchSize, headNum, time, time, true);
//			this.dpreatt = Tensor.createTensor(this.dpreatt, batchSize, headNum, time, time, true);
//		}else {
////			this.dqkv.clearGPU();
////			this.dvaccum.clearGPU();
//		}
//	}
//
//	@Override
//	public void initParam() {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public void output() {
//		// TODO Auto-generated method stub
////		System.out.println("in");
////		this.input.showDM();
//		this.qLayer.forward(this.input);
//		this.kLayer.forward(this.input);
//		this.vLayer.forward(this.input);
//		
//		Tensor query = this.qLayer.getOutput().view(batchSize, time, headNum, dk);
//		Tensor key = this.kLayer.getOutput().view(batchSize, time, headNum, dk);
//		Tensor value = this.vLayer.getOutput().view(batchSize, time, headNum, dk);
//
//		TensorOP.permute(query, qt, new int[] {0, 2, 1, 3});
//		TensorOP.permute(key, kt, new int[] {0, 2, 1, 3});
//		TensorOP.permute(value, vt, new int[] {0, 2, 1, 3});
//		
//		scaledDotProductAttention(qt, kt, vt);
//		
//		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
//		
//		this.oLayer.forward(oi);
//		
//		this.output = this.oLayer.getOutput();
//		
//		if(dropout) {
//			dropoutLayer2.forward(this.oLayer.getOutput());
//			this.output = dropoutLayer2.getOutput();
//		}
//		
//	}
//	
//	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {
//
//		float d_k = (float) (1.0f / Math.sqrt(dk));
//		
////		GPUOP.getInstance().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
//		
//		attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, headNum, time, d_k);
//		
//		Tensor tmp = attn;
//		
//		if(dropout) {
//			dropoutLayer.forward(attn);
//			tmp = dropoutLayer.getOutput();
//		}
//		
////		attn.syncHost();
////		PrintUtils.printImage(attn);
////		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
//
//	}
//
//	public void scaledDotProductAttentionBackward() {
//		
//		Tensor tmp = attn;
//		
//		if(dropout) {
//			tmp = dropoutLayer.getOutput();
//		}
//		
//	    // backward into datt
////		GPUOP.getInstance().bmm(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
//		
//		// backward into dv
////		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);
//		
//		if(dropout) {
//			dropoutLayer.back(dattn);
//			dattn = dropoutLayer.diff;
//		}
//		
//		// backward into preatt
//		float d_k = (float) (1.0f / Math.sqrt(dk));
////		attentionKernel.softmax_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum, d_k);
//		attentionKernel.softmax_unmask_backward(dpreatt, dattn, attn, batchSize, time, headNum, d_k);
//		
//		// backward into q
////		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
//		
//		// backward into k
////		GPUOP.getInstance().bmm(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
//		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
//
//	}
//
//	@Override
//	public Tensor getOutput() {
//		// TODO Auto-generated method stub
//		return output;
//	}
//
//	@Override
//	public void diff() {
//		// TODO Auto-generated method stub
//		
//		if(dropout) {
//			dropoutLayer2.back(delta);
//			this.oLayer.back(dropoutLayer2.diff, oi);
//		}else {
//			this.oLayer.back(delta, oi);
//		}
//
//		attentionKernel.unpermute_backward(dvaccum, oi, batchSize, time, headNum, dk);
//		
//		scaledDotProductAttentionBackward();
//		
//		qt.view(this.qLayer.getOutput().shape());
//		kt.view(this.kLayer.getOutput().shape());
//		vt.view(this.vLayer.getOutput().shape());
//
//		TensorOP.permute(dqt, qt, new int[] {0, 2, 1, 3});
//		TensorOP.permute(dkt, kt, new int[] {0, 2, 1, 3});
//		TensorOP.permute(dvt, vt, new int[] {0, 2, 1, 3});
//
//		Tensor queryDelta = qt.view(batchSize, time, height, width);
//		Tensor keyDelta = kt.view(batchSize, time, height, width);
//		Tensor valueDelta = vt.view(batchSize , time, height, width);
//		
//		this.qLayer.back(queryDelta);
//		this.kLayer.back(keyDelta);
//		this.vLayer.back(valueDelta);
//		
//		TensorOP.add(this.qLayer.diff, this.kLayer.diff, this.qLayer.diff);
//		TensorOP.add(this.qLayer.diff, this.vLayer.diff, this.qLayer.diff);
//
//		this.diff = this.qLayer.diff;
//		
//	}
//
//	@Override
//	public void forward() {
//		// TODO Auto-generated method stub
//		/**
//		 * 参数初始化
//		 */
//		this.init();
//		/**
//		 * 设置输入
//		 */
//		this.setInput();
//		/**
//		 * 计算输出
//		 */
//		this.output();
//	}
//	
//	@Override
//	public void back() {
//		// TODO Auto-generated method stub
//		
//		this.initBack();
//		/**
//		 * 设置梯度
//		 */
//		this.setDelta();
//		/**
//		 * 计算梯度
//		 */
//		this.diff();
//		
//		if(this.network.GRADIENT_CHECK) {
//			this.gradientCheck();
//		}
//
//	}
//
//	@Override
//	public void forward(Tensor input) {
//		// TODO Auto-generated method stub
//		
//		/**
//		 * 参数初始化
//		 */
//		this.init(input);
//		/**
//		 * 设置输入
//		 */
//		this.setInput(input);
//		/**
//		 * 计算输出
//		 */
//		this.output();
//		
//	}
//	
//	@Override
//	public void back(Tensor delta) {
//		// TODO Auto-generated method stub
//
//		this.initBack();
//		/**
//		 * 设置梯度
//		 */
//		this.setDelta(delta);
//		/**
//		 * 计算梯度
//		 */
//		this.diff();
//		
//		if(this.network.GRADIENT_CHECK) {
//			this.gradientCheck();
//		}
//
//	}
//
//	@Override
//	public void update() {
//		// TODO Auto-generated method stub
//		qLayer.update();
//		kLayer.update();
//		vLayer.update();
//		oLayer.update();
//	}
//
//	@Override
//	public void showDiff() {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public LayerType getLayerType() {
//		// TODO Auto-generated method stub
//		return LayerType.mutli_head_attention;
//	}
//
//	@Override
//	public float[][][][] output(float[][][][] input) {
//		// TODO Auto-generated method stub
//		return null;
//	}
//
//	@Override
//	public void initCache() {
//		// TODO Auto-generated method stub
//		
//	}
//
//	@Override
//	public void backTemp() {
//		// TODO Auto-generated method stub
//		
//	}
//	
////	public Tensor getWeights() {
////		return weights;
////	}
//
//	public static void main(String[] args) {
//		
////		int embedDim = 4;
////		int headNum = 2;
////		int batchSize = 3;
////		int time = 3;
////		
////		Transformer tf = new Transformer();
////		tf.number = batchSize * time;
////		tf.time = time;
////		
////		float[] data = RandomUtils.order(batchSize * time * embedDim, 0.1f, 0.1f);
////		
//////		int[] rts = new int[] {2, 3, 3};
//////		
//////		for(int b = 0;b<batchSize;b++) {
//////			int rt = rts[b];
//////			for(int t = 0;t<time;t++) {
//////				if(t > rt) {
//////					for(int n = 0;n<embedDim;n++) {
//////						data[b * time * embedDim + t * embedDim + n] = 0;
//////					}
//////				}
//////			}
//////		}
////		
//////		float[] maskData = new float[] {1,1,1,0,0,1,1,1,1,0,1,1,1,1,0};
//////		
//////		Tensor mask = new Tensor(batchSize, 1, time, time, maskData, true);
////		
////		Tensor input = new Tensor(batchSize * time, 1, 1, embedDim, data, true);
////		
//////		input.showDM();
////		
////		float[] delta_data = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
////		
//////		float[] tmp = MatrixUtils.val(batchSize * time * embedDim, 1.0f);
////		
////		Tensor delta = new Tensor(batchSize * time, 1, 1, embedDim, delta_data, true);
////		
////		DuffsionSelfAttentionLayer mal = new DuffsionSelfAttentionLayer(embedDim, headNum, time, false, false, tf);
////		
//////		mal.forward(input);
////		
////		for(int i = 0;i<10;i++) {
////
////			mal.forward(input);
////			
//////			input.showDM();
////			
//////			mal.getWeights().showDM();
////			
////			mal.getOutput().showShape();
////			
////			mal.getOutput().showDM();
////			
////			mal.back(delta);
//////			delta.showDM();
////			mal.diff.showDM();
//////			delta.copyData(tmp);
////		}
//		
//	}
//	
//	public static boolean same(Tensor a,Tensor b) {
//		float[] ad = a.syncHost();
//		float[] bd = b.syncHost();
//		for(int i=0;i<ad.length;i++) {
//			if(ad[i] != bd[i]) {
//				System.out.println(ad[i]+":"+bd[i] + "["+i+"]");
//				return false;
//			}
//		}
//		return true;
//	}
//	
//}
