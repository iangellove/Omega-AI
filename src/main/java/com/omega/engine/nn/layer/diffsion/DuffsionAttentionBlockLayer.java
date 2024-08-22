package com.omega.engine.nn.layer.diffsion;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.Transformer;
import com.omega.engine.updater.UpdaterFactory;

/**
 * CausalSelfAttentionLayer
 * @author Administrator
 *
 */
public class DuffsionAttentionBlockLayer extends Layer{
	
	private int inChannel = 0;
	
	private int width = 0;
	
	private int height = 0;
	
	private boolean bias = false;
	
	private ConvolutionLayer qLayer;
	private ConvolutionLayer kLayer;
	private ConvolutionLayer vLayer;
	private ConvolutionLayer oLayer;
 
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
	
	private Tensor vaccum;
	
	private Tensor preatt;
	
	private Tensor attn;
	
	private Tensor oi;
	
	private Tensor dvaccum;
	
	private Tensor dattn;
	
	private Tensor dpreatt;
	
	private int batchSize = 1;
	
	private boolean dropout = false;
	
	public DuffsionAttentionBlockLayer(int inChannel,int width,int height,boolean bias,boolean dropout) {
		this.bias = bias;
		this.inChannel = inChannel;
		this.height = height;
		this.width = width;
		this.bias = bias;
		this.oChannel = inChannel;
		this.oHeight = height;
		this.oWidth = width;
		this.dropout = dropout;
		this.initLayers();
	}
	
	public DuffsionAttentionBlockLayer(int inChannel,int width,int height,boolean bias,boolean dropout,Network network) {
		this.bias = bias;
		this.network = network;
		if(this.updater == null) {
			this.setUpdater(UpdaterFactory.create(network.updater, network.updaterParams));
		}
		this.inChannel = inChannel;
		this.height = height;
		this.width = width;
		this.bias = bias;
		this.oChannel = inChannel;
		this.oHeight = height;
		this.oWidth = width;
		this.dropout = dropout;
		this.initLayers();
	}
	
	public void initLayers() {
		this.qLayer = new ConvolutionLayer(inChannel, inChannel, width, height, 1, 1, 0, 1, bias, this.network);
//		this.qLayer.weight = new Tensor(inChannel, inChannel, 1, 1, MatrixUtils.order(inChannel * inChannel, 0.01f, 0.01f), true);
		this.kLayer = new ConvolutionLayer(inChannel, inChannel, width, height, 1, 1, 0, 1, bias, this.network);
//		this.kLayer.weight = new Tensor(inChannel, inChannel, 1, 1, MatrixUtils.order(inChannel * inChannel, 0.01f, 0.01f), true);
		this.vLayer = new ConvolutionLayer(inChannel, inChannel, width, height, 1, 1, 0, 1, bias, this.network);
//		this.vLayer.weight = new Tensor(inChannel, inChannel, 1, 1, MatrixUtils.order(inChannel * inChannel, 0.01f, 0.01f), true);
		
		this.oLayer = new ConvolutionLayer(inChannel, inChannel, width, height, 1, 1, 0, 1, bias, this.network);
//		this.oLayer.weight = new Tensor(inChannel, inChannel, 1, 1, MatrixUtils.order(inChannel * inChannel, 0.01f, 0.01f), true);
		
		if(this.dropout) {
			this.dropoutLayer = new DropoutLayer(0.1f, this.network);
			this.dropoutLayer2 = new DropoutLayer(0.1f, oLayer);
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
		this.number = this.network.number;
		this.batchSize = number;

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
//			this.oi = Tensor.createTensor(this.oi, batchSize, time, headNum, dk, true);
//		}

	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
		this.batchSize = number;
		
		if(this.preatt == null || this.preatt.number != this.batchSize) {
//			System.out.println("in");
			// [batch_size，time，head_num，d_k]
			this.qt = Tensor.createTensor(this.qt, batchSize, height, width, inChannel, true);
			this.kt = Tensor.createTensor(this.vt, batchSize, height, width, inChannel, true);
			this.vt = Tensor.createTensor(this.vt, batchSize, height, width, inChannel, true);
			// [batch_size，n_heads，len_q，len_k]
			this.preatt = Tensor.createTensor(this.preatt, batchSize, width * height, 1, width * height, true);
			// [batch_size，n_heads，len_q，len_k]
			this.attn = Tensor.createTensor(this.attn, batchSize, width * height, 1, width * height, true);
			// [batch_size, n_heads, len_q, dim_v]
			this.vaccum = Tensor.createTensor(this.vaccum, batchSize, width * height, 1, inChannel, true);
			// [batch_size, len_q, n_heads * dim_v]
			this.oi = Tensor.createTensor(this.oi, batchSize, inChannel, 1, height * width, true);
			
			this.output = Tensor.createTensor(this.output, batchSize, inChannel, height, width, true);
		}else {
			this.qt.viewOrg();
			this.vt.viewOrg();
			this.qLayer.getOutput().viewOrg();
			this.kLayer.getOutput().viewOrg();
			this.vLayer.getOutput().viewOrg();
		}

	}
	
	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(this.dvaccum == null){
			this.dvaccum = Tensor.createTensor(this.dvaccum, batchSize, width * height, 1, inChannel, true);
			this.dqt = Tensor.createTensor(this.dqt, batchSize, width * height, 1, inChannel, true);
			this.dkt = Tensor.createTensor(this.dkt, batchSize, width * height, 1, inChannel, true);
			this.dvt = Tensor.createTensor(this.dvt, batchSize, width * height, 1, inChannel, true);
			this.dattn = Tensor.createTensor(this.dattn, batchSize, width * height, 1, width * height, true);
			this.dpreatt = Tensor.createTensor(this.dpreatt, batchSize, width * height, 1, width * height, true);
		}else {
//			this.dqkv.clearGPU();
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
		
		this.qLayer.forward(this.input);
		this.kLayer.forward(this.input);
		this.vLayer.forward(this.input);

		TensorOP.permute(this.qLayer.getOutput(), qt, new int[] {0, 2, 3, 1});
		TensorOP.permute(this.kLayer.getOutput(), kt, new int[] {0, 2, 3, 1});
		TensorOP.permute(this.vLayer.getOutput(), vt, new int[] {0, 2, 3, 1});
		
		qt.view(batchSize, height * width, 1, inChannel);
		kt.view(batchSize, height * width, 1, inChannel);
		vt.view(batchSize, height * width, 1, inChannel);

		scaledDotProductAttention(qt, kt, vt);
		
		TensorOP.permute(vaccum, oi, new int[] {0, 3, 2, 1});
		
		oi.view(batchSize, inChannel, height, width);
		
		this.oLayer.forward(oi);
		
		Tensor tmp = this.oLayer.getOutput();
		
		if(dropout) {
			dropoutLayer2.forward(this.oLayer.getOutput());
			tmp = dropoutLayer2.getOutput();
		}
		
		TensorOP.add(tmp, input, this.output);
//		this.output.showDMByOffset(0, 100);
	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(inChannel));
		
		int m = width * height;
		int n = inChannel;
		int k = inChannel;
		
		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, m, m, n, 1.0f, key.getGpuData(), k, m * k, query.getGpuData(), k, m * k, 0.0f, preatt.getGpuData(), m, m * m, batchSize);

		attentionKernel.softmax_unmask_forward(preatt, attn, batchSize, m, d_k);
		
		Tensor tmp = attn;
		
		if(dropout) {
			dropoutLayer.forward(attn);
			tmp = dropoutLayer.getOutput();
		}
		
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, n, m, m, 1.0f, value.getGpuData(), n, m * k, tmp.getGpuData(), m, m * m, 0.0f, vaccum.getGpuData(), k, m * k, batchSize);

	}

	public void scaledDotProductAttentionBackward(Tensor qt,Tensor key,Tensor vt) {
		
		Tensor tmp = attn;
		
		if(dropout) {
			tmp = dropoutLayer.getOutput();
		}
		
		int m = height * width;
		int n = inChannel;
		int k = inChannel;
	    // backward into datt
		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, m, m, k, 1.0f, vt.getGpuData(), k, m * k, dvaccum.getGpuData(), k, m * k, 0.0f, dattn.getGpuData(), m, m * m, batchSize);
		
		// backward into dv
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, k, m, m, 1.0f, dvaccum.getGpuData(), k, m * k, tmp.getGpuData(), m, m * m, 0.0f, dvt.getGpuData(), k, m * k, batchSize);

		if(dropout) {
			dropoutLayer.back(dattn);
			dattn = dropoutLayer.diff;
		}
		
		// backward into preatt
		float d_k = (float) (1.0f / Math.sqrt(inChannel));
		attentionKernel.softmax_unmask_backward(dpreatt, dattn, attn, batchSize, m, d_k);
		
		// backward into q
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, k, m, m, 1.0f, kt.getGpuData(), k, m * k, dpreatt.getGpuData(), m, m * m, 0.0f, dqt.getGpuData(), k, m * k, batchSize);

		
		// backward into k
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, k, m, m, 1.0f, qt.getGpuData(), k, m * k, dpreatt.getGpuData(), m, m * m, 0.0f, dkt.getGpuData(), k, m * k, batchSize);

	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		if(dropout) {
			dropoutLayer2.back(delta);
			this.oLayer.back(dropoutLayer2.diff, oi);
		}else {
			this.oLayer.back(delta, oi);
		}

		oi.view(batchSize, inChannel, 1, height * width);

		TensorOP.permute(oi, dvaccum, new int[] {0, 3, 2, 1});
		
		scaledDotProductAttentionBackward(qt, kt, vt);
		
		qt.view(batchSize, inChannel, 1, height * width);
		vt.view(batchSize, inChannel, 1, height * width);

		TensorOP.permute(dqt, qt, new int[] {0, 3, 2, 1});
		TensorOP.permute(dkt, kt, new int[] {0, 3, 2, 1});
		TensorOP.permute(dvt, vt, new int[] {0, 3, 2, 1});

		Tensor queryDelta = qt.view(batchSize, inChannel, height, width);
		Tensor keyDelta = kt.view(batchSize, inChannel, height, width);
		Tensor valueDelta = vt.view(batchSize , inChannel, height, width);
		
		this.qLayer.back(queryDelta);
		this.kLayer.back(keyDelta);
		this.vLayer.back(valueDelta);
		
		TensorOP.add(this.qLayer.diff, this.kLayer.diff, this.qLayer.diff);
		TensorOP.add(this.qLayer.diff, this.vLayer.diff, this.qLayer.diff);
		TensorOP.add(this.qLayer.diff, delta, this.qLayer.diff);

		this.diff = this.qLayer.diff;
		
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
		qLayer.update();
		kLayer.update();
		vLayer.update();
		oLayer.update();
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
		
		CUDAModules.initContext();
		
		int N = 2;
		int C = 3;
		int H = 4;
		int W = 4;
		
		float[] data = MatrixUtils.order(N * C * H * W, 0.01f, 0.1f);
		
		Tensor input = new Tensor(N, C, H, W, data, true);
		
		float[] data2 = MatrixUtils.order(N * C * H * W, 0.01f, 0.01f);
		
		Tensor delta = new Tensor(N, C, H, W, data2, true);
		
		Transformer tf = new Transformer();
		
		tf.CUDNN = true;
		tf.number = N;
		
		DuffsionAttentionBlockLayer mal = new DuffsionAttentionBlockLayer(C, W, H, false, false, tf);
		
		mal.forward(input);
		
		mal.getOutput().showDM();
		
		mal.back(delta);
		
		mal.diff.showDM();
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