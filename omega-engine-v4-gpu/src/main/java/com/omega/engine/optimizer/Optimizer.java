package com.omega.engine.optimizer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.LabelUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.check.BaseCheck;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.lr.GDDecay;
import com.omega.engine.optimizer.lr.HalfDecay;
import com.omega.engine.optimizer.lr.LRDecay;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.yolo.model.YoloBox;
import com.omega.yolo.model.YoloDetection;
import com.omega.yolo.utils.BaseDataLoader;
import com.omega.yolo.utils.DetectionDataLoader;
import com.omega.yolo.utils.YoloDataLoader;
import com.omega.yolo.utils.YoloDecode;
import com.omega.yolo.utils.YoloUtils;

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
	
	public BaseCheck check;
	
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
	
	public float min_loss = Float.POSITIVE_INFINITY;
	
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
				if(counter % 10 == 0) {
					this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
				}
				break;
			case SMART_HALF:
				break;
			default:
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
				if(counter % 10 == 0) {
					this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
				}
				break;
			case SMART_HALF:
				if(loss <= min_loss) {
					System.out.println("Validation loss decreased ("+min_loss+" --> "+loss+")");
					min_loss = loss;
					this.counter = 0;
				}else {
					this.counter++;
					System.out.println("Validation loss decreased ("+min_loss+" < "+loss+") update counter:"+this.counter);
				}
				if(this.counter >= 9) {
					this.network.learnRate = HalfDecay.decayedLR(this.network.learnRate);
					this.counter = 0;
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
			
			label.hostToDevice();
			
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
	
	public float testAndLoss(BaseData testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		float error = 0.0f;
		
		float trueCount = 0;
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_DOWN).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			label.hostToDevice();
			
			Tensor output = this.network.predict(input);

			/**
			 * loss
			 */
			Tensor loss = this.network.loss(output, label);
			
			/**
			 * current time error
			 */
			vailLoss += MatrixOperation.sum(loss.syncHost());
			
			output.syncHost();

			int currentError = this.accuracyTrueCount(output, label, testData.labelSet);
			
			trueCount += currentError;
			
//			System.out.println("vaildating["+pageIndex+"] vail accuracy:{"+ currentError * 100 / batchSize +"%} vail loss:{"+currentLoss / batchSize+"}]");
			
		}
		
		error = trueCount / itc / batchSize;
		
		vailLoss = vailLoss / itc / batchSize;
		
		System.out.println("training["+this.trainIndex+"] vail accuracy:{"+ error * 100 +"%} vail loss:{"+vailLoss+"} "+" [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float testAndLoss(BaseDataLoader testData,Tensor input,Tensor label,int batchSize,BaseCheck check) {
		// TODO Auto-generated method stub
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		float accuracy = 0.0f;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			label.hostToDevice();
			
			Tensor output = this.network.predict(input);

			/**
			 * current time error
			 */
			Tensor loss = this.network.loss(output, label);
			
			if(loss.isHasGPU()) {
				vailLoss += MatrixOperation.sum(loss.syncHost());
			}else {
				vailLoss += MatrixOperation.sum(loss.data);
			}
			
			output.syncHost();
			
			accuracy += check.check(output, label, testData.labelSet, true);
			
		}
		
		vailLoss = vailLoss / itc;
		
		System.out.println("test["+this.trainIndex+"] vail loss:{"+vailLoss+"} (accuracy:"+accuracy/testData.number*100+"%) [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float testObjectRecognition(BaseData testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			label.hostToDevice();
			
			Tensor output = this.network.predict(input);

			/**
			 * current time error
			 */
			Tensor loss = this.network.loss(output, label);
			
			if(loss.isHasGPU()) {
				vailLoss += MatrixOperation.sum(loss.syncHost());
			}else {
				vailLoss += MatrixOperation.sum(loss.data);
			}
			
		}
		
		vailLoss = vailLoss / itc / batchSize;
		
		System.out.println("test["+this.trainIndex+"] vail loss:{"+vailLoss+"} [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float testObjectRecognition(BaseDataLoader testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input, label);
			
			input.hostToDevice();
			
			label.hostToDevice();
			
			Tensor output = this.network.predict(input);

			/**
			 * current time error
			 */
			Tensor loss = this.network.loss(output, label);
			
			if(loss.isHasGPU()) {
				vailLoss += MatrixOperation.sum(loss.syncHost());
			}else {
				vailLoss += MatrixOperation.sum(loss.data);
			}
			
		}
		
		vailLoss = vailLoss / itc / batchSize;
		
		System.out.println("test["+this.trainIndex+"] vail loss:{"+vailLoss+"} [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float testObjectRecognition(DetectionDataLoader testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		float vailLoss = 0.0f;
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input, label);
			
			Tensor output = this.network.predict(input);

			/**
			 * current time error
			 */
			Tensor loss = this.network.loss(output, label);
			
			if(loss.isHasGPU()) {
				vailLoss += MatrixOperation.sum(loss.syncHost());
			}else {
				vailLoss += MatrixOperation.sum(loss.data);
			}
			
		}
		
		vailLoss = vailLoss / itc / batchSize;
		
		System.out.println("test["+this.trainIndex+"] vail loss:{"+vailLoss+"} [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return vailLoss;
	}
	
	public float testObjectRecognitionOutputs(BaseData testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Yolo network = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input, label);

			input.hostToDevice();
			
			label.hostToDevice();
			
			Tensor[] output = network.predicts(input);
			
			/**
			 * current time error
			 */
			network.loss(output, label);
			
		}
		
		System.out.println("test["+this.trainIndex+"] [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return 0.0f;
	}
	
	public float testObjectRecognitionOutputs(BaseDataLoader testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Yolo network = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input, label);

			input.hostToDevice();
			
			label.hostToDevice();
			
			if(network.outputNum > 1) {
				Tensor[] output = network.predicts(input);
				/**
				 * current time error
				 */
				network.loss(output, label);
			}else{
				Tensor output = network.predict(input);
				/**
				 * current time error
				 */
				network.loss(output, label);
			}
			
		}
		
		System.out.println("test["+this.trainIndex+"] [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return 0.0f;
	}
	
	public float testObjectRecognitionOutputs(DetectionDataLoader testData,Tensor input,Tensor label,int batchSize) {
		// TODO Auto-generated method stub
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Yolo network = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input, label);

			if(network.outputNum > 1) {
				Tensor[] output = network.predicts(input);
				/**
				 * current time error
				 */
				network.loss(output, label);
				
			}else{
				Tensor output = network.predict(input);
				/**
				 * current time error
				 */
				network.loss(output, label);
			}
			
		}
		
		System.out.println("test["+this.trainIndex+"] [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return 0.0f;
	}
	
	public float testObjectRecognitionOutputs(BaseData testData,int batchSize) {
		// TODO Auto-generated method stub
		
		long startTime = System.nanoTime();
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
		
		Yolo net = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input);

			input.hostToDevice();
			
			Tensor[] output = net.predicts(input);

			for(int l = 0;l<net.outputLayers.size();l++){
				
				YoloLayer layer = (YoloLayer) net.outputLayers.get(l);
				for(int b = 0;b<output[l].number;b++) {
					
					for (int i = 0;i<output[l].height * output[l].width;i++){
				        int row = i / output[l].width;
				        int col = i % output[l].width;
				        for(int n = 0;n<layer.bbox_num;n++){
				        	int n_index = n*output[l].width*output[l].height + row*output[l].width + col;

				            int obj_index = entryIndex(b, output[l].width, output[l].height, n_index, 4, layer.outputs, layer.class_number);
				            float objectness = output[l].data[obj_index];
				            if(objectness > 0.1f) {
				            	 System.out.println(objectness);
				            }
				        }
					}
				}	
			}
			
		}
		
		System.out.println("test["+this.trainIndex+"] [costTime:"+(System.nanoTime()-startTime)/1e6+"ms.]");
		
		return 0.0f;
	}
	
	public static int entryIndex(int batch,int w,int h,int location,int entry,int outputs,int class_number){
	    int n =   location / (w*h);
	    int loc = location % (w*h);
	    return batch*outputs + n*w*h*(4+class_number+1) + entry*w*h + loc;
	}
	
	public float[][][] showObjectRecognition(BaseData testData,Tensor input,int batchSize) {
		// TODO Auto-generated method stub
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input);
			
			input.hostToDevice();
			
			Tensor output = this.network.predict(input);
			
			output.syncHost();
			
			float[][][] draw_bbox = YoloDecode.getDetection(output, testData.width, testData.height);
			
			if((pageIndex + 1) * batchSize > testData.number) {

				System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, (pageIndex + 1) * batchSize - testData.number);
				
			}else {

				System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
				
			}
			
		}
		
		return bbox;
	}
	
	public List<YoloBox> showObjectRecognitionYoloV3(BaseData testData,int batchSize) {
		// TODO Auto-generated method stub
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		List<YoloBox> list = new ArrayList<YoloBox>();
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
		
		Yolo net = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input);
			
			input.hostToDevice();
			
			Tensor[] output = net.predicts(input);
			
			YoloBox[] boxs = new YoloBox[input.number];
			
			for(int i = 0;i<net.outputLayers.size();i++){
				
				YoloLayer layer = (YoloLayer) net.outputLayers.get(i);
				
				YoloDetection[][] dets = YoloUtils.getYoloDetections(output[i], layer.anchors, layer.mask, layer.bbox_num, layer.outputs, layer.class_number, testData.width, testData.height, 0.5f);

				for(int j = 0;j<dets.length;j++) {
					if(boxs[j] != null) {
						boxs[j].getDets().addAll(new ArrayList<>(Arrays.asList(dets[j])));
					}else{
						YoloBox box = new YoloBox(dets[j]);
						boxs[j] = box;
					}
				}
				
			}
			
			list.addAll(new ArrayList<>(Arrays.asList(boxs)));
			
		}
		
		return list;
	}
	
	public List<YoloBox> showObjectRecognitionYoloV3(DetectionDataLoader testData,int batchSize) {
		// TODO Auto-generated method stub
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		List<YoloBox> list = new ArrayList<YoloBox>();
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
		
		Yolo net = (Yolo) this.network;
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.loadData(pageIndex, batchSize, input);
			
			Tensor[] output = net.predicts(input);
			
			YoloBox[] boxs = new YoloBox[input.number];
			
			for(int i = 0;i<net.outputLayers.size();i++){
				
				YoloLayer layer = (YoloLayer) net.outputLayers.get(i);
				
				YoloDetection[][] dets = YoloUtils.getYoloDetections(output[i], layer.anchors, layer.mask, layer.bbox_num, layer.outputs, layer.class_number, this.network.height, this.network.width, 0.5f);

				for(int j = 0;j<dets.length;j++) {
					if(boxs[j] != null) {
						boxs[j].getDets().addAll(new ArrayList<>(Arrays.asList(dets[j])));
					}else{
						YoloBox box = new YoloBox(dets[j]);
						boxs[j] = box;
					}
				}
				
			}
			
			list.addAll(new ArrayList<>(Arrays.asList(boxs)));
			
		}
		
		return list;
	}
	
	public float[][][] showObjectRecognition(BaseData testData,int batchSize) {
		// TODO Auto-generated method stub
		
		this.network.RUN_MODEL = RunModel.TEST;
		
		float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Tensor input = new Tensor(batchSize, testData.channel, testData.height, testData.width, true);
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {

			testData.getBatchData(pageIndex, batchSize, input);
			
			input.hostToDevice();
			
			Tensor output = this.network.predict(input);
			
			output.syncHost();
			
			float[][][] draw_bbox = YoloDecode.getDetection(output, testData.width, testData.height);
			
			if((pageIndex + 1) * batchSize > testData.number) {
				
				/**
				 * 处理不能整除数据
				 * 由于批量预测的时候是向上补充最后一页的数据
				 * 所以在获取bbox的时候需要获取的下标应该是 pageIndex * batchSize - (batchSize - testData.number % batchSize)
				 */
				System.arraycopy(draw_bbox, batchSize - testData.number % batchSize, bbox, pageIndex * batchSize, testData.number % batchSize);
				
			}else {

				System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
				
			}
			
		}
		
		return bbox;
	}
	
	public float[][][] showObjectRecognition(YoloDataLoader testData,int batchSize,int classNum) {
		// TODO Auto-generated method stub
		
		System.out.println("start object recognition.");
		
		long start = System.currentTimeMillis();

		this.network.RUN_MODEL = RunModel.TEST;
		
		float[][][] bbox = new float[testData.number][YoloDecode.grid_size * YoloDecode.grid_size * YoloDecode.bbox_num][YoloDecode.class_number + 1 + 4];
		
		int itc = new BigDecimal(testData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
		
		Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
		
		for(int pageIndex = 0;pageIndex<itc;pageIndex++) {
			
			testData.loadData(pageIndex, batchSize, input);
			
			input.hostToDevice();
			
			Tensor output = this.network.predict(input);
			
			output.syncHost();
			
			float[][][] draw_bbox = YoloDecode.getDetection(output, testData.getDataSet().width, testData.getDataSet().height, classNum);
//			System.out.println(JsonUtils.toJson(draw_bbox));
			if((pageIndex + 1) * batchSize > testData.number) {
				
				/**
				 * 处理不能整除数据
				 * 由于批量预测的时候是向上补充最后一页的数据
				 * 所以在获取bbox的时候需要获取的下标应该是 pageIndex * batchSize - (batchSize - testData.number % batchSize)
				 */
				System.arraycopy(draw_bbox, batchSize - testData.number % batchSize, bbox, pageIndex * batchSize, testData.number % batchSize);
				
			}else {

				System.arraycopy(draw_bbox, 0, bbox, pageIndex * batchSize, batchSize);
				
			}
			
		}
		
		System.out.println("finish object recognition["+((System.currentTimeMillis() - start) / 1000) + "s].");
		
		return bbox;
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
	
	public float testLoss(Tensor output,Tensor labelData) {
		
		float[] data = new float[output.number];
		
		float loss = 0.0f;
		
		for(int n = 0;n<output.number;n++) {
			float onceLoss = testLoss(output.getByNumber(n), labelData.getByNumber(n));
			loss += onceLoss;
			data[n] = onceLoss;
		}
		
		System.out.println("cpu_loss:"+JsonUtils.toJson(data));
		
		return loss;
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
	
	public float testLoss(float[] output,float[] label) {
		
//		System.out.println(JsonUtils.toJson(label));
		
		float sum = 0.0f;
		
		float loss = 0.0f;
		
		/**
		 * max
		 */
		float max = MatrixOperation.max(output);
		
		/**
		 * sum
		 */
		for(int i = 0;i<output.length;i++) {
			sum += Math.exp(output[i] - max);
		}
		
		/**
		 * softmax + log + nlloss
		 */
		for(int i = 0;i<output.length;i++) {
			loss += (float) (-((output[i] - max) - Math.log(sum)) * label[i]);
		}
		return loss;
	}
	
	public static float testLoss2(float[] output,float[] label) {
		
//		System.out.println(JsonUtils.toJson(label));
		
		float sum = 0.0f;
		
		float loss = 0.0f;
		
		/**
		 * max
		 */
		float max = MatrixOperation.max(output);
		
		/**
		 * sum
		 */
		for(int i = 0;i<output.length;i++) {
			sum += Math.exp(output[i] - max);
		}
		
		/**
		 * softmax + log + nlloss
		 */
		for(int i = 0;i<output.length;i++) {
			loss += (float) (-((output[i] - max) - Math.log(sum)) * label[i]);
		}
		return loss;
	}

	public static void main(String[] args) {
		
		float[] x = new float[] {0.6079413f,-1.1546507f,1.444119f,1.5811894f,1.131686f,1.5374337f,0.39088273f,-0.19011068f,-0.010914803f,-1.4776193f};
		
		float[] l = new float[] {0,0,0,1,0,0,0,0,0,0};
		
		System.out.println(testLoss2(x, l));
		
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
