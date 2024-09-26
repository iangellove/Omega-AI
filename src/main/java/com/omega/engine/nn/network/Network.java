package com.omega.engine.nn.network;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.op.TensorOP;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.model.NetworkInit;
import com.omega.engine.updater.UpdaterFactory;
import com.omega.engine.updater.UpdaterType;

import jcuda.Pointer;

/**
 * base network
 * 
 * @author Administrator
 *
 */
public abstract class Network {
	
	public boolean CUDNN = false;
	
	private int threadNum = 8;
	
	public int time;
	
	public NetworkType networkType;
	
	public List<Tensor> paramters = new ArrayList<Tensor>();
	
	public UpdaterType updater = UpdaterType.none;
	
	public RunModel RUN_MODEL = RunModel.TRAIN;
	
	public Map<String,Float> updaterParams;
	
	public boolean GRADIENT_CHECK = false;
	
	public int layerCount = 0;
	
	public int trainingTime = 100;

	public int currentTrainingTime = 0;

	public List<Layer> layerList = new ArrayList<Layer>();
	
	public LossFunction lossFunction;
	
	public Tensor input;

	public Tensor lossDiff;
	
	public float accuracy = 0.01f;
	
	public float learnRate = 0.01f;
	
	public float errorRate = 0.001f;
	
	public float currentError = 0.0f;
	
	/**
	 * batch number
	 */
	public int number = 0;
	
	private int channel = 0;
	
	private int height = 0;
	
	private int width = 0;
	
	public int oChannel = 0;
	
	public int oHeight = 0;
	
	public int oWidth = 0;
	
	public int train_time = 0;

	public long workspaceSize = 0;
	
	public Pointer workspace;
	
	public boolean PROPAGATE_DOWN = false;
	
	public abstract void init() throws Exception;
	
	public abstract Tensor predict(Tensor input);
	
	public abstract Tensor forward(Tensor input);
	
	public abstract void back(Tensor lossDiff);
	
	public abstract Tensor loss(Tensor output,Tensor label);
	
	public abstract Tensor lossDiff(Tensor output,Tensor label);
	
	public abstract Tensor loss(Tensor output,Tensor label, Tensor loss);
	
	public abstract Tensor lossDiff(Tensor output,Tensor label, Tensor diff);
	
	public abstract NetworkType getNetworkType();
	
	public abstract void clearGrad();
	
	public Tensor createParamterGrad(int number,int channel,int height,int width,boolean hasGPU) {
		Tensor pGrad = new Tensor(number, channel, height, width, hasGPU);
		this.addPamamter(pGrad);
		return pGrad;
	}
	
	public void addPamamter(Tensor pGrad) {
		this.paramters.add(pGrad);
	}
	
	public Tensor getDiff() {
//		System.out.println(this.getNextLayer(0).getLayerType().toString());
		return this.getNextLayer(0).diff;
	}
	
	public void setNumber(int number) {
		this.number = number;
	}
	
	public void setInputData(Tensor input) {
		this.number = input.number;
		this.input = input;
	}

	public void setLossDiff(Tensor lossDiff) {
		this.lossDiff = lossDiff;
		this.getLastLayer().setDelta(this.lossDiff);
	}
	
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.getLastLayer().getOutput();
	}
	
	public int getOutputNum() {
		int num = this.getLastLayer().getOutput().channel * this.getLastLayer().getOutput().height * this.getLastLayer().getOutput().width;
		return num;
	}
	
	public Layer getLastLayer() {
		return this.layerList.get(this.layerList.size() - 1);
	}
	
	public Layer getPreLayer(int index) {

		if(index > 0 && index < this.layerCount) {
			return this.layerList.get(index - 1);
		}
		
		return null;
	}
	
	public Layer getNextLayer(int index) {
//		if(index > 0 && index < this.layerCount - 1) {
		if(index < this.layerCount - 1) {
			return this.layerList.get(index + 1);
		}
		return null;
	}
	
	public Layer getLayer(int index) {
		if(index < this.layerCount) {
			return this.layerList.get(index);
		}
		return null;
	}
	
	public Tensor getDelta(int index) {
		
		if(index > 0 && index < this.layerCount - 1) {
			return this.layerList.get(index + 1).diff;
		}else {
			return this.lossDiff;
		}
		
	}
	
	public void addLayer(Layer layer) {
		layer.setNetwork(this);
		layer.setIndex(this.layerList.size());
		if(layer.updater == null) {
			layer.setUpdater(UpdaterFactory.create(this.updater, updaterParams));
		}
		if(layer.index <= 1) {
			layer.PROPAGATE_DOWN = false;
		}
		this.layerList.add(layer);
	}
	
	public NetworkInit save() {
		NetworkInit init = new NetworkInit(this);
		for (Layer layer:this.layerList) {
			init.getLayers().add(layer.save());
		}
		return init;
	}
	
	public void update() {
		
		this.train_time += 1;
		
		for(int i = layerCount - 1;i>=0;i--) {
			
			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;
//			System.out.println(layer.getLayerType());
			layer.update();
			
		}	
		
	}
	
	public void update(int count) {
		
		this.train_time += 1;

		for(int i = layerCount - 1;i>=0;i--) {
			
			Layer layer = layerList.get(i);
			
			layer.learnRate = this.learnRate;

			layer.update();
			
		}	
		
	}
	
	public void unfreeze() {
		for(int i = 0;i<layerCount;i++) {
			layerList.get(i).freeze = false;
		}	
	}
	
	public void saveToJson(String path) {
		
	}

	public int getThreadNum() {
		return threadNum;
	}

	public void setThreadNum(int threadNum) {
		this.threadNum = threadNum;
	}
	
	public void clipGradNorm(float clip) {
		float total_norm = 0.0f;
		float[] norms = new float[paramters.size()];
		System.out.println(paramters.size());
		for(int i = 0;i<paramters.size();i++) {
			Tensor pGrad = paramters.get(i);
			norms[i] = pGrad.norm();
//			total_norm += Math.pow(pGrad.norm(), 2.0d);
		}
		total_norm = MatrixOperation.norm(norms);
		total_norm = (float) Math.pow(total_norm, (1.0d / 2.0d));
		float clip_coef = clip / (total_norm + 1e-6f);
		System.out.println("clip_coef:"+clip_coef);
		if(clip_coef < 1) {
			for(Tensor pGrad:paramters) {
				TensorOP.mul(pGrad, clip_coef, pGrad);
//				pGrad.showDM();
			}
		}
	}

	public int getChannel() {
		return channel;
	}

	public void setChannel(int channel) {
		this.channel = channel;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}
	
}
