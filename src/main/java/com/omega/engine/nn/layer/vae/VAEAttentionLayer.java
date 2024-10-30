package com.omega.engine.nn.layer.vae;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.DropoutLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.gpu.AttentionKernel;
import com.omega.engine.nn.layer.normalization.GNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.UpdaterFactory;

/**
 * CausalSelfAttentionLayer
 * @author Administrator
 *
 */
public class VAEAttentionLayer extends Layer{
	
	private int groups = 0;
	
	private int time;
	
	private int headNum = 1;
	
	private int embedDim = 0;
	
	private int dk = 0;
	
	private boolean bias = false;
	
	private boolean residualConnect = false;
	
	private float rescaleOutput = 1.0f;
	
	private GNLayer gn;
	
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
	
	public VAEAttentionLayer(int embedDim,int headNum,int time,float rescaleOutput,boolean bias,boolean dropout,boolean residualConnect) {
		this.bias = bias;
		this.rescaleOutput = rescaleOutput;
		this.residualConnect = residualConnect;
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
	
	public VAEAttentionLayer(int embedDim,int headNum,int time,float rescaleOutput,boolean bias,boolean dropout,boolean residualConnect,Network network) {
		this.bias = bias;
		this.rescaleOutput = rescaleOutput;
		this.residualConnect = residualConnect;
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
	
	public VAEAttentionLayer(int embedDim,int headNum,int time,int groups,float rescaleOutput,boolean bias,boolean dropout,boolean residualConnect,Network network) {
		this.bias = bias;
		this.groups = groups;
		this.rescaleOutput = rescaleOutput;
		this.residualConnect = residualConnect;
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
		
		if(groups > 0) {
			gn = new GNLayer(groups, this);
		}
		
		this.setqLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

		this.setkLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

		this.setvLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

		this.setoLinerLayer(new FullyLayer(embedDim, embedDim, bias, this.network));

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
			this.dqt = Tensor.createGPUTensor(this.dqt, batchSize, headNum, time, dk, true);
			this.dkt = Tensor.createGPUTensor(this.dkt, batchSize, headNum, time, dk, true);
			this.dvt = Tensor.createGPUTensor(this.dvt, batchSize, headNum, time, dk, true);
			this.dattn = Tensor.createGPUTensor(this.dattn, batchSize, headNum, time, time, true);

		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		Tensor x = this.input;
		
		if(gn != null) {
			gn.forward(x);
			x = gn.getOutput();
		}
		
		this.getqLinerLayer().forward(x);
		this.getkLinerLayer().forward(x);
		this.getvLinerLayer().forward(x);
		
		Tensor query = this.getqLinerLayer().getOutput().view(batchSize, time, headNum, dk);
		Tensor key = this.getkLinerLayer().getOutput().view(batchSize, time, headNum, dk);
		Tensor value = this.getvLinerLayer().getOutput().view(batchSize, time, headNum, dk);

		TensorOP.permute(query, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(key, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(value, vt, new int[] {0, 2, 1, 3});
		
		scaledDotProductAttention(qt, kt, vt);

		Tensor vaccum = temp;
		attentionKernel.unpermute(vaccum, oi, batchSize, time, headNum, dk);
		
		this.getoLinerLayer().forward(oi);
		
		this.output = this.getoLinerLayer().getOutput();
		
		if(dropout) {
			dropoutLayer2.forward(this.getoLinerLayer().getOutput());
			this.output = dropoutLayer2.getOutput();
		}
		
		if(residualConnect) {
			TensorOP.add(this.output, this.input, this.output);
		}
		
		if(rescaleOutput != 1.0f){
			TensorOP.div(this.output, rescaleOutput, this.output);
		}

	}
	
	public void scaledDotProductAttention(Tensor query,Tensor key,Tensor value) {

		float d_k = (float) (1.0f / Math.sqrt(dk));
		
		Tensor preatt = temp;
		
		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, key.getGpuData(), dk, time * dk, query.getGpuData(), dk, time * dk, 0.0f, preatt.getGpuData(), time, time * time, batchSize * headNum);
		
		if(network.RUN_MODEL == RunModel.TEST) {
			attentionKernel.scale(preatt, d_k, batchSize, headNum, time);
			attentionKernel.softmax_test_forward(preatt, attn, batchSize, headNum, time);
		}else {
			attentionKernel.softmax_forward(preatt, attn, batchSize, headNum, time, d_k);
		}

		Tensor tmp = attn;
		
		if(dropout) {
			dropoutLayer.forward(attn);
			tmp = dropoutLayer.getOutput();
		}
//		value.showDM();
		Tensor vaccum = temp;
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, value.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, vaccum.getGpuData(), dk, time * dk, batchSize * headNum);
	}

	public void scaledDotProductAttentionBackward() {
		
		Tensor tmp = attn;
		
		if(dropout) {
			tmp = dropoutLayer.getOutput();
		}
		Tensor dvaccum = temp;
	    // backward into datt
		GPUOP.getInstance().bmmEX(CUBLAS_OP_T, CUBLAS_OP_N, time, time, dk, 1.0f, vt.getGpuData(), dk, time * dk, dvaccum.getGpuData(), dk, time * dk, 0.0f, dattn.getGpuData(), time, time * time, batchSize * headNum);
		
		// backward into dv
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, dvaccum.getGpuData(), dk, time * dk, tmp.getGpuData(), time, time * time, 0.0f, dvt.getGpuData(), dk, time * dk, batchSize * headNum);

		if(dropout) {
			dropoutLayer.back(dattn);
			dattn = dropoutLayer.diff;
		}
		
		// backward into preatt
		float d_k = (float) (1.0f / Math.sqrt(dk));
//		attentionKernel.softmax_backward(dpreatt, dattn, attn, batchSize, time, embedDim, headNum, d_k);
		attentionKernel.softmax2_backward(dattn, attn, batchSize, time, embedDim, headNum, d_k);
		Tensor dpreatt = dattn;
//		System.err.println("dpreatt:");
//		System.err.println("error:"+MatrixUtils.check(dpreatt.syncHost(), dpreatt2.syncHost()));
//		dpreatt.checkDMZero();
		// backward into q
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_N, dk, time, time, 1.0f, kt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dqt.getGpuData(), dk, time * dk, batchSize * headNum);
		
		// backward into k
		GPUOP.getInstance().bmmEX(CUBLAS_OP_N, CUBLAS_OP_T, dk, time, time, 1.0f, qt.getGpuData(), dk, time * dk, dpreatt.getGpuData(), time, time * time, 0.0f, dkt.getGpuData(), dk, time * dk, batchSize * headNum);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		if(rescaleOutput != 1.0f){
			TensorOP.div(this.delta, rescaleOutput, this.delta);
		}
		
		if(dropout) {
			dropoutLayer2.back(delta);
			this.getoLinerLayer().back(dropoutLayer2.diff, oi);
		}else {
			this.getoLinerLayer().back(delta, oi);
		}
		
		attentionKernel.unpermute_backward(temp, oi, batchSize, time, headNum, dk);
		
		scaledDotProductAttentionBackward();

		qt.view(this.getqLinerLayer().getOutput().shape());
		kt.view(this.getqLinerLayer().getOutput().shape());
		vt.view(this.getqLinerLayer().getOutput().shape());
		
		TensorOP.permute(dqt, qt, new int[] {0, 2, 1, 3});
		TensorOP.permute(dkt, kt, new int[] {0, 2, 1, 3});
		TensorOP.permute(dvt, vt, new int[] {0, 2, 1, 3});

		Tensor queryDelta = qt.view(batchSize * time, 1, 1, headNum * dk);
		Tensor keyDelta = kt.view(batchSize * time, 1, 1, headNum * dk);
		Tensor valueDelta = vt.view(batchSize * time, 1, 1, headNum * dk);
		
		this.getqLinerLayer().back(queryDelta);
		this.getkLinerLayer().back(keyDelta);
		this.getvLinerLayer().back(valueDelta);
		
		TensorOP.add(this.getqLinerLayer().diff, this.getkLinerLayer().diff, this.getqLinerLayer().diff);
		TensorOP.add(this.getqLinerLayer().diff, this.getvLinerLayer().diff, this.getqLinerLayer().diff);
		
		this.diff = this.getqLinerLayer().diff;
		
		if(gn != null) {
			gn.back(this.diff);
			this.diff = gn.diff;
		}
		
		if(residualConnect) {
			TensorOP.add(this.diff, this.delta, this.diff);
		}

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
