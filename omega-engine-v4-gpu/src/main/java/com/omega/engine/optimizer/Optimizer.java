package com.omega.engine.optimizer;

import java.math.BigDecimal;

import com.omega.common.data.Tensor;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.lr.GDDecay;
import com.omega.engine.optimizer.lr.HalfDecay;
import com.omega.engine.optimizer.lr.LRDecay;
import com.omega.engine.optimizer.lr.LearnRateUpdate;

/**
 * 
 * Optimizer
 * 
 * @author Administrator
 *
 */
public abstract class Optimizer {
	
	private String sid;
	
	public int batchIndex = 1;
	
	public int trainIndex = 1;
	
	public int batchSize = 1;
	
	public int dataSize = 0;
	
	public Tensor loss;
	
	public Tensor lossDiff;
	
	public int trainTime = 10;
	
	public int minTrainTime = 10000;
	
	public float currentError = 1.0f;
	
	public float error = 0.01f;
	
	public Network network;
	
	private TaskEngine trainEngine;
	
	public LearnRateUpdate learnRateUpdate = LearnRateUpdate.NONE;
	
	private BaseData trainingData;
	
	private BaseData testData;
	
	private boolean warmUp = false;
	
	public int burnIn = 300;
	
	public int power = 4;
	
	public float scale = 0.1f;
	
	public int step = 500;
	
	public float gama = 0.9999f;
	
	public float lr = 0.1f;
	
	public boolean isOnline = false;
	
	public int lrStartTime = 5;
	
	public float max_lr = 0.1f;
	
	public float min_loss = Float.NEGATIVE_INFINITY;
	
	public int counter = 0;
	
	public abstract void train(BaseData trainingData);
	
	public abstract void train(BaseData trainingData,BaseData testData);
	
	public abstract void train(BaseData trainingData,BaseData validata,BaseData testData);
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,float error,boolean warmUp) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.error = error;
		this.lr = network.learnRate;
		this.max_lr = network.learnRate;
		this.warmUp = warmUp;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,float error,boolean warmUp) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.warmUp = warmUp;
		this.lr = network.learnRate;
		this.max_lr = network.learnRate;
		this.network.init();
	}
	
	/**
	 * 
	 * @param network
	 * @param trainTime
	 * @param error
	 * @throws Exception 
	 */
	public Optimizer(Network network,int batchSize,int trainTime,int minTrainTime,float error,boolean warmUp,LearnRateUpdate learnRateUpdate) throws Exception {
		this.network = network;
		this.trainTime = trainTime;
		this.minTrainTime = minTrainTime;
		this.error = error;
		this.warmUp = warmUp;
		this.learnRateUpdate = learnRateUpdate;
		this.lr = network.learnRate;
		this.max_lr = network.learnRate;
		this.network.init();
	}
	
	public void setTrainEngine(TaskEngine trainEngine) {
		this.trainEngine = trainEngine;
	}
	
	public TaskEngine getTrainEngine() {
		return this.trainEngine;
	}

	public BaseData getTrainingData() {
		return trainingData;
	}

	public void setTrainingData(BaseData trainingData) {
		this.trainingData = trainingData;
	}

	public BaseData getTestData() {
		return testData;
	}

	public void setTestData(BaseData testData) {
		this.testData = testData;
	}
	
	public void updateLR() {
		
		if(warmUp && batchIndex < burnIn) {
			this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
		}else {

			switch (this.learnRateUpdate) {
			case LR_DECAY:
				this.network.learnRate = LRDecay.decayedLR(this.max_lr, this.network.learnRate, this.trainIndex, 5);
				break;
			case GD_GECAY:
				this.network.learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
				break;
			case NONE:
				break;
			case CONSTANT:
				break;
			case COSINE:
				if(this.trainIndex >= lrStartTime) {
					this.network.learnRate = (float) (0.5d * max_lr * (Math.cos(this.trainIndex/trainTime * Math.PI)) + 1.0d) * this.network.learnRate;
				}else {
					this.network.learnRate = this.trainIndex * this.network.learnRate / lrStartTime;
				}
				break;
			case RANDOM:
				this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
				break;
			case POLY:
				float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
				this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
				break;
			case STEP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
				break;
			case EXP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
				break;
			case SIG:
				this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
				break;
			case HALF:
				this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate, this.trainIndex, 10);
				break;
			}
			
		}
		
	}
	
	public void updateLR(float loss) {
		
		if(warmUp && batchIndex < burnIn) {
			this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
		}else {
			
			switch (this.learnRateUpdate) {
			case LR_DECAY:
				this.network.learnRate = LRDecay.decayedLR(this.max_lr, this.network.learnRate, this.trainIndex, 5);
				break;
			case GD_GECAY:
				this.network.learnRate = GDDecay.decayedLR(this.max_lr, this.trainIndex);
				break;
			case NONE:
				break;
			case CONSTANT:
				break;
			case COSINE:
				if(this.trainIndex >= lrStartTime) {
					this.network.learnRate = (float) (0.5d * max_lr * (Math.cos(this.trainIndex/trainTime * Math.PI)) + 1.0d) * this.network.learnRate;
				}else {
					this.network.learnRate = this.trainIndex * this.network.learnRate / lrStartTime;
				}
				break;
			case RANDOM:
				this.network.learnRate = (float) Math.pow(RandomUtils.getInstance().nextFloat(), power) * this.lr;
				break;
			case POLY:
				float t = batchIndex * 1.0f / trainTime / dataSize * batchSize;
				this.network.learnRate = (float) (this.lr * Math.pow((1.0f - t), power));
				break;
			case STEP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.scale, batchIndex / step));
				break;
			case EXP:
				this.network.learnRate = (float) (this.lr * Math.pow(this.gama, batchIndex));
				break;
			case SIG:
				this.network.learnRate = (float) (this.lr / (1.0f + Math.pow(Math.E, this.gama * (batchIndex - step))));
				break;
			case HALF:
				this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate, this.trainIndex, 10);
				break;
			case SMART_HALF:
				if(loss <= min_loss) {
					this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate, this.counter, 10);
					min_loss = loss;
					this.counter = 0;
				}else {
					this.counter++;
				}
				break;
			}
			
		}
		
	}
	
	public float test(BaseData testData) {
		// TODO Auto-generated method stub
		float error = 0.0f;
		
		float trueCount = 0;
		
		Tensor sample = new Tensor(1, testData.channel, testData.height, testData.width, true);

		for(int n = 0;n<testData.number;n++) {
			
//			float[] onceError = MatrixOperation.subtraction(output, testData.dataLabel[i]);
			
			testData.getOnceData(n, sample);

			Tensor output = this.network.predict(sample);
			
			output.syncHost();
			
			String label = testData.labels[n];
			
//			System.out.println(JsonUtils.toJson(output.data));
			
			String predictLabel = LabelUtils.vectorTolabel(output.data, testData.labelSet);
			
//			System.out.println("final output:"+JsonUtils.toJson(output.data));
//			
//			System.out.println(label+"="+predictLabel+":"+label.equals(predictLabel));
			
			if(!label.equals(predictLabel)) {
//				System.out.println("index:"+n+"::"+JsonUtils.toJson(output)+"==>predictLabel:"+predictLabel+"==label:"+label+":"+label.equals(predictLabel));
			}else {
				trueCount++;
			}
			
		}
		
		error = trueCount / testData.number;
		
		System.out.println("准确率:"+ error * 100 +"%");
		
		return error;
	}
	
	public float test(BaseData testData,int batchSize) {
		// TODO Auto-generated method stub
		float error = 0.0f;
		
		float trueCount = 0;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
		
		Tensor label = new Tensor(batchSize, testData.label.channel, testData.label.height, testData.label.width);
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			Tensor output = this.network.predict(input);
			
			output.syncHost();
			
			trueCount += this.accuracyTrueCount(output, label, testData.labelSet);

		}
		
		error = trueCount / itc / batchSize;
		
		System.out.println("training["+this.trainIndex+"] vail accuracy:{"+ error * 100 +"%}"+" [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return error;
	}
	
	public float testAndLoss(BaseData testData,int batchSize) {
		// TODO Auto-generated method stub
		float error = 0.0f;
		
		float trueCount = 0;
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
		
		Tensor label = new Tensor(batchSize, testData.label.channel, testData.label.height, testData.label.width);
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			Tensor output = this.network.predict(input);
			
			/**
			 * loss
			 */
			Tensor loss = this.network.loss(output, label);
			
			/**
			 * current time error
			 */
			vailLoss += MatrixOperation.sum(loss.syncHost()) / batchSize;
			
			output.syncHost();
			
			trueCount += this.accuracyTrueCount(output, label, testData.labelSet);

		}
		
		error = trueCount / itc / batchSize;
		
		vailLoss = vailLoss / itc;
		
		System.out.println("training["+this.trainIndex+"] vail accuracy:{"+ error * 100 +"%} vail loss:{"+vailLoss+"} "+" [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float accuracy(Tensor output,Tensor labelData,String[] labelSet) {
		
		float error = 0.0f;
		float trueCount = 0;

		for(int n = 0;n<output.number;n++) {
			
			String label = LabelUtils.vectorTolabel(labelData.getByNumber(n), labelSet);
			
			String predictLabel = LabelUtils.vectorTolabel(output.getByNumber(n), labelSet);
			
			if(label.equals(predictLabel)) {
				trueCount++;
			}
			
		}
		
		error = trueCount / output.number * 100;

		return error;
	}
	
	public int accuracyTrueCount(Tensor output,Tensor labelData,String[] labelSet) {

		int trueCount = 0;

		for(int n = 0;n<output.number;n++) {
			
			String label = LabelUtils.vectorTolabel(labelData.getByNumber(n), labelSet);
			
			String predictLabel = LabelUtils.vectorTolabel(output.getByNumber(n), labelSet);
//			System.out.println(label+":"+predictLabel);
			if(label.equals(predictLabel)) {
				trueCount++;
			}
			
		}
		
		return trueCount;
	}

	public boolean isWarmUp() {
		return warmUp;
	}

	public void setWarmUp(boolean warmUp) {
		this.warmUp = warmUp;
	}
	
	public void online(boolean isOnline) {
		this.isOnline = isOnline;
	}

	public String getSid() {
		return sid;
	}

	public void setSid(String sid) {
		this.sid = sid;
	}
	
}