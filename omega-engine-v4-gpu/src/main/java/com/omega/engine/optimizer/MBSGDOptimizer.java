package com.omega.engine.optimizer;

import com.omega.common.data.Tensor;
import com.omega.common.data.utils.DataTransforms;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.controller.TrainTask;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.OutputsNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.yolo.utils.BaseDataLoader;
import com.omega.yolo.utils.DetectionDataLoader;
import com.omega.yolo.utils.YoloLabelUtils;

import jcuda.driver.JCudaDriver;

/**
 * 
 * Mini Batch Stochastic Gradient Descent
 * 
 * @author Administrator
 *
 */
public class MBSGDOptimizer extends Optimizer {
	
	private YoloLabelUtils u;
	
	public YoloLabelUtils dataEnhanceInstance() {
		if(u == null) {
			u = new YoloLabelUtils(1, 4);
		}
		return u;
	}
	
	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}

	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	public MBSGDOptimizer(String sid,Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.setSid(sid);
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
	}
	
	@Override
	public void train(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

//				int[][] indexs = MathUtils.sortInt(trainingData.number,this.batchSize);
				
//				int[][] indexs = new int[468][128];
//				
//				DataExportUtils.importTXT(indexs, "H://index3.txt");
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
//				for(int it = 0;it<1;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

					/**
					 * current time error
					 */
					this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					
//					long back_start = System.nanoTime();
					
					lossDiff.hostToDevice();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					output.syncHost();
					
//					JCudaDriver.cuCtxSynchronize();
					
//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
					
					float error = this.accuracy(output, label, trainingData.labelSet);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					/**
					 * 发送消息
					 */
					if(isOnline && this.getSid() != null) {
						
						TrainTask.sendMsg(this.getSid(), msg);
						
					}
					
//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();

			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

	@Override
	public void train(BaseData trainingData, BaseData testData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

//				int[][] indexs = MathUtils.sortInt(trainingData.number,this.batchSize);
				
//				int[][] indexs = new int[468][128];
//				
//				DataExportUtils.importTXT(indexs, "H://index3.txt");
				
				this.network.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
//				for(int it = 0;it<1;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();

//					this.loss.clear();
//					
//					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(label.syncHost()));
//					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
//					
//					System.out.println(JsonUtils.toJson(this.lossDiff.syncHost()));
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

//					long back_start = System.nanoTime();
					
//					loss.hostToDevice();
					
//					lossDiff.hostToDevice();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					JCudaDriver.cuCtxSynchronize();
					
//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");

					output.syncHost();
					
					float error = this.accuracy(output, label, trainingData.labelSet);

					/**
					 * current time error
					 */
					this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					/**
					 * 发送消息
					 */
					if(isOnline && this.getSid() != null) {
						
						TrainTask.sendMsg(this.getSid(), msg);
						
					}
					
//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				/**
				 * vail data test
				 */
				this.test(testData, this.batchSize);
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}

	@Override
	public void train(BaseData trainingData, BaseData validata, BaseData testData) {
		// TODO Auto-generated method stub
		
	}
	
	public void train(BaseData trainingData, BaseData validata, float[] mean, float[] std) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor transData = new Tensor(trainingData.number, trainingData.channel, trainingData.height, trainingData.width);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);

			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				transforms(trainingData.input, transData, mean, std);
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}

					trainingData.randomData(indexs[it], transData.data, input, label);

					input.hostToDevice();
					
					label.hostToDevice();

					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();

					output.syncHost();

					float error = this.accuracy(output, label, trainingData.labelSet);

					/**
					 * current time error
					 */
					this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					
					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					/**
					 * 发送消息
					 */
					if(isOnline && this.getSid() != null) {
						
						TrainTask.sendMsg(this.getSid(), msg);
						
					}

					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * vail data test
				 */
				float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);

				/**
				 * update learning rate
				 */
				this.updateLR(vail_loss);
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void train(BaseDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
//			/**
//			 * normalize vailSet
//			 */
//			DataTransforms.normalize(validata.input, mean, std);

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				float train_loss = 0.0f;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {

					long start = System.nanoTime();

					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					trainingData.loadData(indexs[it], input, label);
					
//					System.out.println(JsonUtils.toJson(label.data));
					
					input.hostToDevice();
					
					label.hostToDevice();

					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();

//					output.syncHost();

//					float error = this.accuracy(output, label, trainingData.labelSet);
					
					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					/**
					 * 发送消息
					 */
					if(isOnline && this.getSid() != null) {
						
						TrainTask.sendMsg(this.getSid(), msg);
						
					}

					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
//				/**
//				 * vail data test
//				 */
//				float vail_loss = this.testAndLoss(validata, vail_input, vail_label, this.batchSize);

				/**
				 * update learning rate
				 */
				this.updateLR();
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

//				int[][] indexs = MathUtils.sortInt(trainingData.number,this.batchSize);
				
//				int[][] indexs = new int[468][128];
//				
//				DataExportUtils.importTXT(indexs, "H://index3.txt");
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
//				for(int it = 0;it<1;it++) {
					
//					if(Math.abs(this.currentError) <= this.error) {
//						break;
//					}
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 

					input.hostToDevice();
					
					label.hostToDevice();
					
//					input.showDM();
					
//					long output_start = System.nanoTime();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);
					
//					System.out.println(JsonUtils.toJson(output.data));
//					System.out.println("output1:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
//					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(output.data));
					
//					System.out.println("output2:"+(System.nanoTime() - output_start) / 1e6 + "ms.");
					
					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
//					System.out.println("=========>:"+JsonUtils.toJson(lossDiff.data));

//					long back_start = System.nanoTime();
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
//					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
					
//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
					
//					float error = 0.0f;
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
//			System.out.println(JsonUtils.toJson(this.network.layerList));
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData,BaseData validata) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, validata.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label);

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 200 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(validata, vail_input, vail_label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(BaseData trainingData,BaseData validata,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, validata.channel, validata.height, validata.width, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYolo(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					Tensor output = this.network.forward(input);

					/**
					 * loss
					 */
					this.loss = this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = this.network.lossDiff(output, label);
					
					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}
//					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:"+this.currentError+" [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 200 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(validata, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognition(DetectionDataLoader trainingData,DetectionDataLoader valiData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			Yolo network = (Yolo) this.network;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = trainingData.initLabelTensor();
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					Tensor output = network.forward(input);
					
					/**
					 * loss
					 */
					this.network.loss(output, label);
					
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label);
					
					/**
					 * back
					 */
					network.back(lossDiff);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 200 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognition(valiData, input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(BaseData trainingData,BaseData valiData,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize);
			
			Tensor vail_input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);

				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.getRandomData(indexs[it], input, label); 
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYoloV3(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
					
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);

					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 200 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(BaseDataLoader trainingData,BaseDataLoader valiData,boolean dataEnhance) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor vail_label = new Tensor(batchSize, 1, 1, valiData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.loadData(indexs[it], input, label);
					
					/**
					 * 数据增强
					 */
					if(dataEnhance) {
						dataEnhanceInstance().transforms(input, label);
						YoloLabelUtils.formatToYolo(label, input.height, input.width);
					}

					input.hostToDevice();
					
					label.hostToDevice();
					
					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
					System.out.println("in--------------->");
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);
					
					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 100 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, vail_label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void trainObjectRecognitionOutputs(DetectionDataLoader trainingData,DetectionDataLoader valiData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();
			
			OutputsNetwork network = (OutputsNetwork) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			Tensor label = trainingData.initLabelTensor();
			
			Tensor vail_input = new Tensor(batchSize, this.network.channel, this.network.height, this.network.width, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.network.RUN_MODEL = RunModel.TRAIN;
				
				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					long start = System.nanoTime();

					this.loss.clear();
					
					this.lossDiff.clear();
					
					trainingData.loadData(indexs[it], input, label);

					/**
					 * forward
					 */
					network.forward(input);
					
					/**
					 * loss
					 */
					network.loss(label);
//					System.out.println("in--------------->");
					/**
					 * loss diff
					 */
					Tensor[] lossDiffs = network.lossDiff(label);
					
					/**
					 * back
					 */
					network.back(lossDiffs);
					
					/**
					 * update
					 */
					this.network.update();
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR();
				
				if(this.learnRateUpdate == LearnRateUpdate.SMART_HALF && this.trainIndex % 200 == 0) {
					
					this.network.learnRate = this.network.learnRate * 0.5f;
					
				}
				
				if(this.trainIndex % 100 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testObjectRecognitionOutputs(valiData, vail_input, label, this.batchSize);
					
					System.out.println("----------------testing finish---------------");
					
				}
				
			}
			
			/**
			 * 停止训练
			 */
			System.out.println("training finish. ["+this.trainIndex+"] finalError:"+this.currentError);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public void transforms(Tensor trainData,Tensor transData, float[] mean,float[] std){
		
		/**
		 * 随机裁剪
		 */
		DataTransforms.randomCrop(trainData, transData, 32, 32, 4);
		
		/**
		 * 随机翻转
		 */
		DataTransforms.randomHorizontalFilp(transData, transData);
		
		/**
		 * normalize
		 */
		DataTransforms.normalize(transData, transData, mean, std);

		/**
		 * cutcout
		 */
		DataTransforms.cutout(transData, transData, 16, 1);
		
		System.out.println("data transform finish.");
		
	}
	
	public void transforms2(Tensor trainData,Tensor transData, float[] mean,float[] std){
		
		/**
		 * normalize
		 */
		DataTransforms.normalize(trainData, transData, mean, std);

		System.out.println("data transform finish.");
		
	}

}
