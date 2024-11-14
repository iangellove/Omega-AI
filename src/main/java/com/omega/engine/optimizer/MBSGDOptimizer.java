package com.omega.engine.optimizer;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.data.utils.DataTransforms;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.check.BaseCheck;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.grad.GradClipping;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.network.DiffusionUNet;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.OutputsNetwork;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.nn.network.vae.TinyVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.nn.network.vae.VQVAE;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.rnn.data.OneHotDataLoader;
import com.omega.example.rnn.data.RNNDataLoader;
import com.omega.example.yolo.data.BaseDataLoader;
import com.omega.example.yolo.data.DetectionDataLoader;
import com.omega.example.yolo.utils.YoloLabelUtils;

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
	
	public MBSGDOptimizer(Network network, int trainTime, float error,int batchSize,LearnRateUpdate learnRateUpdate,boolean warmUp,BaseCheck check) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
		this.batchSize = batchSize;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.learnRateUpdate = learnRateUpdate;
		this.check = check;
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
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
					if(this.loss.isHasGPU()){
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

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

//					System.out.println("back:"+(System.nanoTime() - back_start) / 1e6 + "ms.");
					
					float error = this.accuracy(output, label, trainingData.labelSet);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} currentError:"+this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);

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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				this.network.RUN_MODEL = RunModel.TRAIN;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
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
					
//					/**
//					 * update learning rate
//					 */
//					this.updateLR();
					
					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
					
//					input.hostToDevice();
//					
//					label.hostToDevice();

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
				this.updateLR(this.lr_step);
				
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
	
	public void train(BaseDataLoader trainingData,BaseDataLoader valiData,BaseCheck check) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
					Tensor loss = this.network.loss(output, label);
					
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
					
					if(loss.isHasGPU()) {
						loss.syncHost();
					}
					
					float accuracy = check.check(output, label, trainingData.labelSet, false);
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") (loss:"+loss.getByIndex(0, 0, 0, 0)+") (accuracy:"+accuracy/batchSize*100+"%) [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(this.trainIndex % 10 == 0) {
					
					System.out.println("----------------testing start----------------");
					
					this.testAndLoss(valiData, input, label, this.batchSize, check);
					
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
	
	public void trainObjectRecognition(BaseData trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize);
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, 1, trainingData.labelSize, true);
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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
				this.updateLR(this.lr_step);
				
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
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = trainingData.initLabelTensor();
			
			Tensor vail_input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				if(this.trainIndex == 2) {
					this.network.unfreeze();
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
				this.updateLR(this.lr_step);
				
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
	
	public void testRNN(Tensor input) {
		try {
			
			CUDAModules.initCUDAFunctions();
			
			/**
			 * forward
			 */
			Tensor output = this.network.forward(input);
			
			output.showDM();
			
			/**
			 * loss diff
			 */
			float[] ld = MatrixUtils.one(output.dataLength);
			this.lossDiff = new Tensor(output.number, output.channel, output.height, output.width, ld, true);

			/**
			 * back
			 */
			this.network.back(this.lossDiff);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public void trainRNN(RNNDataLoader trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();
			
			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(trainingData.time * batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
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

//					System.out.println(output2TXT(input, trainingData));
					
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
					this.lossDiff = network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
					/**
					 * grad clipping
					 */
//					this.gradClipping(this.network);
					
					/**
					 * update
					 */
					this.network.update();
					
					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
					}

//					train_loss += this.currentError;
					
					output.syncHost();
					
					float error = this.accuracy(output, label);
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
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
	
	public void trainSeg(BaseDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor label = new Tensor(batchSize, 1, this.network.getHeight(), this.network.getWidth(), true);
			
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
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
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
	
	public static void q_mean_variance(Tensor x_0,Tensor x_t,Tensor t,float[] posterior_mean_coef1,float[] posterior_mean_coef2,float[] posterior_mean) {
		
		for(int b = 0;b<x_t.number;b++) {
			for(int i = 0;i<x_t.getOnceSize();i++) {
				int idx = b * x_t.getOnceSize() + i;
				posterior_mean[idx] = posterior_mean_coef1[b] * x_0.data[idx] - posterior_mean_coef2[b] * x_t.data[idx];
			}
		}
		
		
	}
	
	public Map<String,float[]> initGussianDiffusionTest(Tensor x_t,int T,float beta_1,float beta_T) {
		
		Map<String,float[]> result = new HashMap<String, float[]>();
		
		float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);

		float[] alphas = MatrixOperation.subtraction(1, betas);

		float[] alphas_bar = MatrixUtils.cumprod(alphas);

		float[] alphas_bar_prev = new float[alphas_bar.length];
		alphas_bar_prev[0] = 1.0f;
		for(int i = 1;i<alphas_bar_prev.length;i++) {
			alphas_bar_prev[i] = alphas_bar[i - 1];
		}

		float[] sqrt_recip_alphas_bar = new float[alphas_bar.length];
		float[] sqrt_recipm1_alphas_bar = new float[alphas_bar.length];
		float[] posterior_var = new float[alphas_bar.length];
		float[] posterior_log_var_clipped = new float[alphas_bar.length];
		float[] posterior_mean_coef1 = new float[alphas_bar.length];
		float[] posterior_mean_coef2 = new float[alphas_bar.length];
		float[] model_log_var  = new float[alphas_bar.length];
		
		for(int i = 0;i<alphas_bar.length;i++) {
			sqrt_recip_alphas_bar[i] = (float) Math.sqrt(1 / alphas_bar[i]);
			sqrt_recipm1_alphas_bar[i] = (float) Math.sqrt(1 / alphas_bar[i] - 1);
			posterior_var[i] = betas[i] * (1 - alphas_bar_prev[i]) / (1 - alphas_bar[i]);
			if(i == 0) {
				posterior_log_var_clipped[i] = (float) Math.log(betas[1] * (1 - alphas_bar_prev[1]) / (1 - alphas_bar[1]));
			}else {
				posterior_log_var_clipped[i] = (float) Math.log(posterior_var[i]);
			}
			posterior_mean_coef1[i] = (float) (Math.sqrt(alphas_bar_prev[i]) * betas[i] / (1 - alphas_bar[i]));
			posterior_mean_coef2[i] = (float) (Math.sqrt(alphas[i]) * (1 - alphas_bar_prev[i]) / (1 - alphas_bar[i]));
			if(i == 0) {
				model_log_var[i] = (float) Math.log(betas[1] * (1 - alphas_bar_prev[1]) / (1 - alphas_bar[1]));
			}else {
				model_log_var[i] = (float) Math.log(betas[i]);
			}
		}
		
		float[] posterior_mean = new float[x_t.dataLength];
		
		result.put("model_log_var", model_log_var);
		result.put("sqrt_recip_alphas_bar", sqrt_recip_alphas_bar);
		result.put("sqrt_recipm1_alphas_bar", sqrt_recipm1_alphas_bar);
		result.put("posterior_mean_coef1", posterior_mean_coef1);
		result.put("posterior_mean_coef2", posterior_mean_coef2);
		result.put("posterior_mean", posterior_mean);
		
		return result;
	}
	
	public void testGaussianDiffusion(Tensor x_t,Tensor t,int T,float beta_1,float beta_T,Map<String,float[]> params,float[] mean,float[] std) {
		
		try {
			
			DiffusionUNet network = (DiffusionUNet) this.network;

			RandomUtils.gaussianRandom(x_t);
			
			float[] model_log_var = params.get("model_log_var");
			float[] sqrt_recip_alphas_bar = params.get("sqrt_recip_alphas_bar");
			float[] sqrt_recipm1_alphas_bar = params.get("sqrt_recipm1_alphas_bar");
			float[] posterior_mean_coef1 = params.get("posterior_mean_coef1");
			float[] posterior_mean_coef2 = params.get("posterior_mean_coef2");
			float[] posterior_mean = params.get("posterior_mean");
			
			for(int timestep = T - 1;timestep>=0;timestep--) {
				
				int[] t_data = MatrixUtils.valInt(x_t.number, timestep);
				
				float[] model_log_var_t = MatrixUtils.gather(model_log_var, t_data);
				
				float[] exsa1 = MatrixUtils.gather(sqrt_recip_alphas_bar, t_data);
				
				float[] exsa2 = MatrixUtils.gather(sqrt_recipm1_alphas_bar, t_data);
				
				float[] posterior_mean_coef1_t  = MatrixUtils.gather(posterior_mean_coef1, t_data);
				
				float[] posterior_mean_coef2_t  = MatrixUtils.gather(posterior_mean_coef2, t_data);
				
//				float[] posterior_log_var_clipped_t = MatrixUtils.gather(posterior_log_var_clipped, t_data);

				t.setData(t_data);
				
				Tensor eps = network.forward(x_t, t);
				
				predict_xstart_from_eps(x_t, t, eps, exsa1, exsa2, posterior_mean_coef1_t, posterior_mean_coef2_t, posterior_mean);

				decodeXT(posterior_mean, model_log_var_t, x_t, timestep);
				
				System.out.println(timestep);
			}
			
			MatrixOperation.clampSelf(x_t.data, -1, 1);
			
			/**
			 * print image
			 */
			showImgs("H:\\voc\\gan_anime\\duffsion_test\\", x_t);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void decodeXT(float[] mean,float[] log_var,Tensor x_t,int it) {
		for(int b = 0;b<x_t.number;b++) {
			for(int i = 0;i<x_t.getOnceSize();i++) {
				int idx = b * x_t.getOnceSize() + i;
				if(it == 0) {
					x_t.data[idx] = mean[idx];
				}else {
					x_t.data[idx] = (float) (mean[idx] + Math.exp(0.5 * log_var[b]) * RandomUtils.randomGaussianFloat());
				}
			}
		}
		x_t.hostToDevice();
	}
	
	public void predict_xstart_from_eps(Tensor x_t,Tensor t,Tensor eps,float[] sqrt_recip_alphas_bar,float[] sqrt_recipm1_alphas_bar,
			float[] posterior_mean_coef1,float[] posterior_mean_coef2,float[] posterior_mean) {
		
		float[] eps_data = eps.syncHost();
		
		for(int b = 0;b<x_t.number;b++) {
			for(int i = 0;i<x_t.getOnceSize();i++) {
				int idx = b * x_t.getOnceSize() + i;
				float x_0 = sqrt_recip_alphas_bar[b] * x_t.data[idx] - sqrt_recipm1_alphas_bar[b] * eps_data[idx];
				posterior_mean[idx] = posterior_mean_coef1[b] * x_0 - posterior_mean_coef2[b] * x_t.data[idx];
			}
		}
		
	}
	

//	public static void main(String[] args) {
//		
//		try {
//			
//			int N = 2;
//			int C = 3;
//			int H = 4;
//			int W = 4;
//			int T = 10;
//			float beta_1 = 1e-4f;
//			float beta_T = 0.02f;
//			
//			Tensor noiseInput = new Tensor(N, C, H, W, true);
//			
//			Tensor t = new Tensor(N, 1, 1, 1, true);
//			
//			testGaussianDiffusion(noiseInput, t, T, beta_1, beta_T);
//			
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//		}
//		
//	}
	
//	public void testGaussianDiffusion(String it,int ddim_timesteps,Tensor noiseInput,Tensor noise) {
//		
//		try {
//			
//			DuffsionUNet network = (DuffsionUNet) this.network;
//			
//			float beta_1 = 1e-4f;
//			float beta_T = 0.02f;
//			int T = 1000;
//			float[] mean = new float[] {0.5f, 0.5f, 0.5f};
//			float[] std = new float[] {0.5f, 0.5f, 0.5f};
//			
////			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
//			
//			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
//			
////			RandomUtils.gaussianRandom(noiseInput);
//			
//			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
//			float[] alphas = MatrixOperation.subtraction(1, betas);
//			float[] alphas_bar = MatrixUtils.cumprod(alphas);
//			
//			int step = T / ddim_timesteps;
//			
//			float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
//			
//			float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
//			
//			for(int i = 1;i<ddim_timestep_seq.length;i++) {
//				ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
//			}
//			int[] t_data = new int[batchSize];
//			int[] prev_t_data = new int[batchSize];
//			for(int timestep = ddim_timesteps - 1;timestep>=0;timestep--) {
//				for(int i = 0;i<batchSize;i++) {
//					t_data[i] = (int) ddim_timestep_seq[timestep];
//					prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
//				}
//				t.setData(t_data);
//				
//				Tensor eps = noise;
////				eps.showDMByOffset(0, 100);
//				float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
//				
//				float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);
//				
//				prev_mean_from_eps(noiseInput, eps, exsa1, exsa2, 1, timestep);
//				
//				noiseInput.hostToDevice();
//				
//				if(timestep == 100) {
//					MatrixOperation.clampSelf(noiseInput.data, -1, 1);
//					
//					/**
//					 * print image
//					 */
//					showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it+"_100");
//				}
//				
//			}
//			
//			MatrixOperation.clampSelf(noiseInput.data, -1, 1);
//			
//			/**
//			 * print image
//			 */
//			showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it);
//			
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//		}
//		
//	}
	

	public void prev_mean_from_eps(Tensor xt,Tensor t,float[] alphas_bar,float[] alphas_bar_prev,float eta,int timestep) {
		
		DiffusionUNet network = (DiffusionUNet) this.network;
//		xt.showDMByOffset(0, 100);
		Tensor eps = network.forward(xt, t);
		float[] eps_data = eps.syncHost();
		float[] noise = RandomUtils.gaussianRandom(eps.dataLength, 1.0f);
//		System.out.println(JsonUtils.toJson(noise));
//		xt.syncHost();
//		eps.showDMByOffset(0, 96);
		for(int b = 0;b<xt.number;b++) {
			float sigma_t = (float) (eta * Math.sqrt((1.0f - alphas_bar_prev[b]) / (1.0f - alphas_bar[b]) * (1.0f - alphas_bar[b] / alphas_bar_prev[b])));
			for(int l = 0;l<xt.getOnceSize();l++) {
				int i = b * xt.getOnceSize() + l;
				float pred_x0 = (float) ((xt.data[i] - Math.sqrt(1.0f - alphas_bar[b]) * eps_data[i]) / Math.sqrt(alphas_bar[b]));
				if(pred_x0 > 1) {
					pred_x0 = 1;
				}else if(pred_x0 < -1){
					pred_x0 = -1;
				}
				float pred_dir_xt = (float) (Math.sqrt(1.0f - alphas_bar_prev[b] - sigma_t * sigma_t) * eps_data[i]);

				xt.data[i] =  (float)Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t * noise[i];
//				xt.data[i] = (float) (Math.sqrt(alphas_bar_prev[b]) * pred_x0 + pred_dir_xt + sigma_t);
			}
		}
		xt.hostToDevice();
//		xt.showDMByOffset(0, 100);
	}

	public void testGaussianDiffusion(String it,int ddim_timesteps,Tensor noiseInput,Tensor t) {
		
		try {
			
			float beta_1 = 1e-4f;
			float beta_T = 0.02f;
			int T = 1000;
			float[] mean = new float[] {0.5f, 0.5f, 0.5f};
			float[] std = new float[] {0.5f, 0.5f, 0.5f};
			
//			Tensor noiseInput = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
//			
//			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
			
//			RandomUtils.gaussianRandom2(noiseInput, 0, 1);
			RandomUtils.gaussianRandom(noiseInput, 0, 1);
			
			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
			float[] alphas = MatrixOperation.subtraction(1, betas);
			float[] alphas_bar = MatrixUtils.cumprod(alphas);
			
			int step = T / ddim_timesteps;
			
			float[] ddim_timestep_seq = MatrixUtils.range(0, T, step, 1);
			
			float[] ddim_timestep_prev_seq = new float[ddim_timestep_seq.length];
			
			for(int i = 1;i<ddim_timestep_seq.length;i++) {
				ddim_timestep_prev_seq[i] = ddim_timestep_seq[i - 1];
			}
			int[] t_data = new int[batchSize];
			int[] prev_t_data = new int[batchSize];

			for(int timestep = ddim_timesteps - 1;timestep>=0;timestep--) {
				for(int i = 0;i<batchSize;i++) {
					t.data[i] = ddim_timestep_seq[timestep];
					t_data[i] = (int) ddim_timestep_seq[timestep];
					prev_t_data[i] = (int) ddim_timestep_prev_seq[timestep];
				}
				t.hostToDevice();
				
				float[] exsa1 = MatrixUtils.gather(alphas_bar, t_data);
				
				float[] exsa2 = MatrixUtils.gather(alphas_bar, prev_t_data);

				prev_mean_from_eps(noiseInput, t, exsa1, exsa2, 1, timestep);

			}
			
			noiseInput.data = MatrixOperation.clampSelf(noiseInput.data, -1, 1);
			
			/**
			 * print image
			 */
			showImgs("H:\\voc\\gan_anime\\duffsion_test\\", noiseInput, it, mean, std);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void showImgs(String outputPath,Tensor input) {

		ImageUtils utils = new ImageUtils();
		
		if(input.isHasGPU()) {
			input.syncHost();
		}
		
		for(int b = 0;b<input.number;b++) {
			float[] once = input.getByNumber(b);
//			once = MatrixOperation.add(once, 0.5f);
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true), input.height, input.width, null, null);
		}
		
	}
	
	public static void showImgs(String outputPath,Tensor input,String it,float[] mean,float[] std) {

		ImageUtils utils = new ImageUtils();

		for(int b = 0;b<input.number;b++) {
			float[] once = input.getByNumber(b);
			utils.createRGBImage(outputPath + it+ "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
		}
		
	}
	
	public static void showImgs(String outputPath,Tensor input,String it) {

		ImageUtils utils = new ImageUtils();
		
//		if(input.isHasGPU()) {
//			input.syncHost();
//		}
		
		for(int b = 0;b<input.number;b++) {
			float[] once = input.getByNumber(b);
//			once = MatrixOperation.add(once, 0.5f);
			utils.createRGBImage(outputPath + it+ "_" + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true), input.height, input.width, null, null);
		}
		
	}
	
	public static void showImgs(String outputPath,Tensor input,float[] mean,float[] std) {

		ImageUtils utils = new ImageUtils();
		
		if(input.isHasGPU()) {
			input.syncHost();
		}

		for(int b = 0;b<input.number;b++) {
			float[] once = input.getByNumber(b);
//			once = MatrixOperation.add(once, 0.5f);
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.channel, input.height, input.width, true, mean, std), input.height, input.width, null, null);
		}
		
	}
	
	public void trainGaussianDiffusion(DiffusionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			DiffusionUNet network = (DiffusionUNet) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			float beta_1 = 1e-4f;
			float beta_T = 0.02f;
			int T = 1000;
			
			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
//			Tensor x_t = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			Tensor t = new Tensor(batchSize, 1, 1, 1, true);
			
			Tensor noise = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			float[] betas = MatrixUtils.linspace(beta_1, beta_T, T);
			float[] alphas = MatrixOperation.subtraction(1, betas);
			float[] alphas_bar = MatrixUtils.cumprod(alphas);
			float[] sqrt_alphas_bar = MatrixOperation.sqrt(alphas_bar);
			float[] sqrt_one_minus_alphas_bar = MatrixOperation.sqrt(MatrixOperation.subtraction(1, alphas_bar));
			
//			Map<String,float[]> testParams = initGussianDiffusionTest(input, T, beta_1, beta_T);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
//				int[][] indexs = trainingData.order();

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
					
//					int[] t_data = RandomUtils.randomInt2(0, T - 1, batchSize);
					int[] t_data = RandomUtils.randomInt(0, T - 1, batchSize);
//					int[] t_data = new int[] {100, 902, 31, 698};
					
//					System.out.println(JsonUtils.toJson(t_data));
					t.setData(t_data);
//					t.showDM();
					float[] exsa1 = MatrixUtils.gather(sqrt_alphas_bar, t_data);
					
					float[] exsa2 = MatrixUtils.gather(sqrt_one_minus_alphas_bar, t_data);
					
					trainingData.loadData(indexs[it], exsa1, exsa2, input, noise);
					
					JCudaDriver.cuCtxSynchronize();
					
//					/**
//					 * print image
//					 */
//					if(it > 0 && it % 1 == 0) {
//						float[] mean = new float[] {0.5f, 0.5f, 0.5f};
//						float[] std = new float[] {0.5f, 0.5f, 0.5f};
//						showImgs("E:\\voc\\gan_anime\\duffsion_test_input\\", input, mean, std);
//					}
					
					/**
					 * forward
					 */
					Tensor output = network.forward(input, t);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, noise);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, noise);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / this.batchSize;
//						System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / this.batchSize;
					}

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
					
					if(it > 0 && it % 500 == 0) {
						network.RUN_MODEL = RunModel.TEST;
						System.out.println("start create test images.");
//						testGaussianDiffusion(i + "_" + it, 200, input, noise);
						testGaussianDiffusion(i + "_" + it, 200, input, t);
						System.out.println("finish create.");
//						testGaussianDiffusion(x_t, t, T, beta_1, beta_T, testParams, trainingData.mean, trainingData.std);
						network.RUN_MODEL = RunModel.TRAIN;
//						this.network.learnRate = this.network.learnRate * 0.1f;
					}
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
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
	
	public void trainTinyVAE(DiffusionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			TinyVAE network = (TinyVAE) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
//				int[][] indexs = trainingData.order();

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
					
					trainingData.loadData(indexs[it], input);
					
					JCudaDriver.cuCtxSynchronize();

					/**
					 * forward
					 */
					Tensor output = network.forward(input);
					
					/**
					 * loss
					 */
					float loss = network.totalLoss(output, input);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, input);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					this.currentError = loss;

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;

					/**
					 * update learning rate
					 */
//					this.updateLR(this.lr_step);
//					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(i % 10 == 0) {

					/**
					 * showImage
					 */
					this.network.RUN_MODEL = RunModel.TEST;
					
					Tensor output = network.forward(input);
					output.syncHost();
//					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
					
					/**
					 * print image
					 */
					showImgs("H:\\vae_dataset\\pokemon-blip\\test\\", output, i + "", trainingData.mean, trainingData.std);
					
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
	
	public void trainVQVAE(DiffusionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			VQVAE network = (VQVAE) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
//				int[][] indexs = trainingData.order();

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
					
					trainingData.loadData(indexs[it], input);
					
					JCudaDriver.cuCtxSynchronize();

					/**
					 * forward
					 */
					Tensor output = network.forward(input);
					
					/**
					 * loss
					 */
					float loss = network.totalLoss(output, input);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, input);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					this.currentError = loss;

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;

					/**
					 * update learning rate
					 */
//					this.updateLR(this.lr_step);
//					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it);
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
				this.updateLR(this.lr_step);
				
				if(i % 1 == 0) {

					/**
					 * showImage
					 */
					this.network.RUN_MODEL = RunModel.TEST;
					
					Tensor output = network.forward(input);
					output.syncHost();
//					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
					
					/**
					 * print image
					 */
					showImgs("H:\\vae_dataset\\pokemon-blip\\test128\\", output, i + "", trainingData.mean, trainingData.std);
					
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
	
	public void trainTinyVQVAE(DiffusionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			TinyVQVAE network = (TinyVQVAE) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
//				int[][] indexs = trainingData.order();

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
					
					trainingData.loadData(indexs[it], input);
					
					JCudaDriver.cuCtxSynchronize();

					/**
					 * forward
					 */
					Tensor output = network.forward(input);

					/**
					 * loss
					 */
					float loss = network.totalLoss(output, input);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, input);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					this.currentError = loss;

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
					
					/**
					 * update learning rate
					 */
					this.updateLR(this.lr_step);
					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);

					if(it % 100 == 0) {

						/**
						 * showImage
						 */
						this.network.RUN_MODEL = RunModel.TEST;
						
						output = network.forward(input);
						output.syncHost();
//						output.data = MatrixOperation.clampSelf(output.data, -1, 1);
						
						/**
						 * print image
						 */
						showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
						
						this.network.RUN_MODEL = RunModel.TRAIN;
						
					}
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
//				this.updateLR(this.lr_step);
				
//				if(i % 10 == 0) {
//
//					/**
//					 * showImage
//					 */
//					this.network.RUN_MODEL = RunModel.TEST;
//					
//					Tensor output = network.forward(input);
//					output.syncHost();
////					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
//					
//					/**
//					 * print image
//					 */
//					showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
//					
//				}
				
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
	
	public void trainTinyVQVAE2(DiffusionImageDataLoader trainingData) {
		// TODO Auto-generated method stub

		try {
			
			CUDAModules.initCUDAFunctions();

			TinyVQVAE2 network = (TinyVQVAE2) this.network;
			
			this.dataSize = trainingData.number;

			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}

			Tensor input = new Tensor(batchSize, this.network.getChannel(), this.network.getHeight(), this.network.getWidth(), true);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}

				this.trainIndex = i + 1;
				
				int[][] indexs = trainingData.shuffle();
//				int[][] indexs = trainingData.order();

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
					
					trainingData.loadData(indexs[it], input);
					
					JCudaDriver.cuCtxSynchronize();

					/**
					 * forward
					 */
					Tensor output = network.forward(input);

					/**
					 * loss
					 */
					float loss = network.totalLoss(output, input);

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, input);

					/**
					 * back
					 */
					network.back(this.lossDiff);
//					System.out.println(JsonUtils.toJson(this.loss.syncHost()));
					/**
					 * update
					 */
					network.update();

					JCudaDriver.cuCtxSynchronize();
					
					/**
					 * current time error
					 */
					this.currentError = loss;

					train_loss += this.currentError;
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;
					
					/**
					 * update learning rate
					 */
					this.updateLR(this.lr_step);
					updateLRDynamic(i * trainingData.count_it + it, this.trainTime * trainingData.count_it, 1e-6f);

//					if(it % 100 == 0) {
//
//						/**
//						 * showImage
//						 */
//						this.network.RUN_MODEL = RunModel.TEST;
//						
//						output = network.forward(input);
//						output.syncHost();
////						output.data = MatrixOperation.clampSelf(output.data, -1, 1);
//						
//						/**
//						 * print image
//						 */
//						showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
//						
//						this.network.RUN_MODEL = RunModel.TRAIN;
//						
//					}
					
				}
				
				System.out.println("training["+this.trainIndex+"] train loss:{"+train_loss/indexs.length+"} ");
				
				/**
				 * update learning rate
				 */
//				this.updateLR(this.lr_step);
				
				if(i % 1 == 0) {

					/**
					 * showImage
					 */
					this.network.RUN_MODEL = RunModel.TEST;
					
					Tensor output = network.forward(input);
					output.syncHost();
					output.data = MatrixOperation.clampSelf(output.data, -1, 1);
					
					/**
					 * print image
					 */
					showImgs("H:\\vae_dataset\\pokemon-blip\\test256\\", output, i + "", trainingData.mean, trainingData.std);
					
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
	
	public void updateLRDynamic(int it,int count,float min) {
		int warmup_iters = 0;
		int lr_decay_iters = count;
//		System.out.println(this.lr);
//		System.out.println(lr_decay_iters);
	    double min_lr = min;
		
	    if (it < warmup_iters){
	    	network.learnRate = this.lr * it / warmup_iters;
	        return;
	    }
	    if(it > lr_decay_iters) {
	    	network.learnRate = (float) min_lr;
	    	return;
	    }
	    BigDecimal decay_ratio = new BigDecimal(0);
	    
	    if(it > 0) {
	    	decay_ratio = new BigDecimal(it - warmup_iters).divide(new BigDecimal(lr_decay_iters - warmup_iters), 24, BigDecimal.ROUND_HALF_DOWN);
	    }
//	    System.out.println(decay_ratio.doubleValue());
	    
	    BigDecimal coeff = new BigDecimal(0.5d).multiply(new BigDecimal(1).add(new BigDecimal(Math.cos(new BigDecimal(Math.PI).multiply(decay_ratio).doubleValue()))));
	    
	    BigDecimal tlr = new BigDecimal(min_lr).add(coeff.multiply(new BigDecimal((this.lr - min_lr))));
	    tlr = tlr.setScale(24, BigDecimal.ROUND_HALF_DOWN);

	    network.learnRate = (float)tlr.doubleValue();
	}
	
	public static String output2TXT(Tensor output,RNNDataLoader trainData) {
		String txt = "";
//		output.showDMByNumber(0);
		OneHotDataLoader tr = (OneHotDataLoader) trainData;
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 1);
			char c = tr.dictionaryData[charIndex];
			txt += c;
		}
		return txt;
	}
	
	public static int pickTopN(float[] x,int n) {

		float[] sort = Arrays.copyOf(x, x.length);
		
		Arrays.sort(sort);

		float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);

		float v = topN[RandomUtils.getRandomNumber(topN)];
		
		for(int i = 0;i<x.length;i++) {
			if(v == x[i]) {
				return i;
			}
		}
		
		return 0;
	}
	
	public void gradClipping(Network network) {
		
		for(Layer layer:network.layerList) {
			if(layer.diffW != null) {
//				System.out.println(layer.getLayerType()+"-diffW");
				GradClipping.gradClipping(layer.diffW, 1e-7f);
			}
			if(layer.diffB != null) {
//				System.out.println("diffB");
				GradClipping.gradClipping(layer.diffB, 1e-7f);
			}
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
