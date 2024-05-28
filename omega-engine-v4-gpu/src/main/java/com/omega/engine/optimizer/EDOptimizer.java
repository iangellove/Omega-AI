package com.omega.engine.optimizer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.nn.data.BaseData;
import com.omega.engine.nn.grad.GradClipping;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.network.GPT;
import com.omega.engine.nn.network.GPT2;
import com.omega.engine.nn.network.NanoGPT;
import com.omega.engine.nn.network.Network;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.Seq2Seq;
import com.omega.engine.nn.network.Seq2SeqRNN;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.rnn.data.IndexDataLoader;
import com.omega.transformer.utils.CNChatTokenizer;
import com.omega.transformer.utils.CNChatTokenizer2;
import com.omega.transformer.utils.CNTokenizer;
import com.omega.transformer.utils.ENTokenizer;

import jcuda.driver.JCudaDriver;

public class EDOptimizer extends Optimizer {

	private float clamp_val = -100;
	
	public EDOptimizer(Network network, int batchSize, int trainTime, float error,LearnRateUpdate learnRateUpdate, boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		this.batchSize = batchSize;
		this.trainTime = trainTime;
		this.learnRateUpdate = learnRateUpdate;
		this.loss = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
		this.lossDiff = new Tensor(batchSize, this.network.oChannel, this.network.oHeight, this.network.oWidth);
	}
	
	public EDOptimizer(Network network, int batchSize, int trainTime, float error, boolean warmUp) throws Exception {
		super(network, batchSize, trainTime, error, warmUp);
		// TODO Auto-generated constructor stub
	}
	
	@Override
	public void train(BaseData trainingData) {
		// TODO Auto-generated method stub
		
	}
	
	public void trainSeq2SeqRNN(IndexDataLoader trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Seq2SeqRNN network = (Seq2SeqRNN) this.network;
			
			Tensor inputEncoder = new Tensor(network.en_time * batchSize, 1, 1, network.en_len, true);

			Tensor inputDecoder = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
			
			Tensor label = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);

			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], inputEncoder, inputDecoder, label);
					
					/**
					 * forward
					 */
					output = network.forward(inputEncoder, inputDecoder);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, label);
					
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
					
//					/**
//					 * grad clipping
//					 */
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
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / inputDecoder.number;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / inputDecoder.number;
					}

//					train_loss += this.currentError;
					
					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracy(output, label, time, batchSize);
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainSeq2Seq(IndexDataLoader trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			Seq2Seq network = (Seq2Seq) this.network;
			
			Tensor inputEncoder = new Tensor(network.en_time * batchSize, 1, 1, network.en_len, true);

			Tensor inputDecoder = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);
			
			Tensor label = new Tensor(network.de_time * batchSize, 1, 1, network.de_len, true);

			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], inputEncoder, inputDecoder, label);
					
					/**
					 * forward
					 */
					output = network.forward(inputEncoder, inputDecoder);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, label);
					
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
					
//					/**
//					 * grad clipping
//					 */
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
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / inputDecoder.number;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / inputDecoder.number;
					}

//					train_loss += this.currentError;
					
					output.syncHost();
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracy(output, label, time, batchSize);
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainGPT(ENTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			GPT network = (GPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
			
			Tensor mask = ENTokenizer.triu(batchSize, network.head_num, network.time, network.time, 1);
			
			Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
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
					output = network.forward(input, positions, mask);
//					output.showDMByNumber(0);
//					label.showDMByNumber(0);
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainGPT2(ENTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			GPT2 network = (GPT2) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
			Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);
			
			Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
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
					output = network.forward(input, positions, mask);
//					output.showDMByNumber(0);
//					label.showDMByNumber(0);
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
//					this.gradClipping(this.network);
					
					this.network.clipGradNorm(1.0f);
					
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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainGPT(CNChatTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			GPT network = (GPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocab_size, true);
			
			Tensor mask = ENTokenizer.triu(batchSize, network.head_num, network.time, network.time, 1);
			
			Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
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
					output = network.forward(input, positions, mask);
//					output.showDMByNumber(0);
//					label.showDMByNumber(0);
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
//					lossDiff.showDMByNumber(0);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainGPT2(CNChatTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			GPT2 network = (GPT2) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
			Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);
			
			Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
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
					output = network.forward(input, positions, mask);
//					output.showDMByNumber(0);
//					label.showDMByNumber(0);
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
//					lossDiff.showDMByNumber(0);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);
					
					this.batchIndex++;

				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainNanoGPT(ENTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			NanoGPT network = (NanoGPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
			Tensor mask = ENTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);
			
			Tensor positions = ENTokenizer.getPositions(batchSize, network.time);
			
			int itc = new BigDecimal(trainingData.number).divide(new BigDecimal(batchSize), 0, BigDecimal.ROUND_UP).intValue();
			
			int[][] tmp = new int[itc][batchSize];
			
			List<Integer> list = new ArrayList<Integer>(); 
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.number,this.batchSize, tmp, list);
				
				Tensor output = null;
				
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
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadData(indexs[it], input, label);
					
					
					/**
					 * forward
					 */
					output = network.forward(input, positions, mask);
//					output.showDMByNumber(0);
//					label.showDMByNumber(0);
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
//					this.gradClipping(this.network);
					
//					this.network.clipGradNorm(1.0f);
					
					/**
					 * update
					 */
					this.network.update();
					
//					JCudaDriver.cuCtxSynchronize();
					
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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
				}
//				output.showDMByNumber(0);
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);
				
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
	
	public void trainNanoGPT(CNChatTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			NanoGPT network = (NanoGPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
//			Tensor mask = CNChatTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);

			Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;

				int[][] indexs = MathUtils.randomInts(trainingData.trainData.size(),this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadTrainData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					output = network.forward(input, positions);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
//					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
//					this.lossDiff = network.lossDiff(output, label);
//					lossDiff.showDMByNumber(0);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
//					this.gradClipping(this.network);
					
//					this.network.clipGradNorm(1.0f);
					
					/**
					 * update
					 */
//					if(it<indexs.length - 1){
						this.network.update();
//					}

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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
					
					if(it != 0 && it % 200 == 0) {
						vail_chat(network, input, output, label, positions, trainingData);
						network.RUN_MODEL = RunModel.TRAIN;
					}
					
//					if(it != 0 && it % 500 == 0) {
//						break;
//					}
					
				}
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);

//				output.showDMByNumber(1);
//				
//				OPKernel.getInstance().copy_number_gpu(input, input1, 0, 0);
//				
//				output1 = network.forward(input1, positions1, mask1);
//				output1.showDMByNumber(1);
//				
//				System.out.println("==============>");
//				GPTTest.output2TXT(input1, trainingData, true);
//				GPTTest.output2TXT(output1, trainingData, true);
//				input1.showDMByNumber(0);
				
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
	
	public void trainNanoGPT(CNChatTokenizer2 trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			NanoGPT network = (NanoGPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
//			Tensor mask = CNChatTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);

			Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;

				int[][] indexs = MathUtils.randomInts(trainingData.trainData.size(),this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
					long start = System.nanoTime();
					
					this.loss.clear();

					this.lossDiff.clear();
					
					/**
					 * 读取训练数据
					 */
					trainingData.loadTrainData(indexs[it], input, label);
					
					/**
					 * forward
					 */
					output = network.forward(input, positions);
					
					/**
					 * loss
					 */
					this.loss = network.loss(output, label, trainingData.tokenizer.specials.get("<pad>"));

					/**
					 * loss diff
					 */
					this.lossDiff = network.lossDiff(output, label, trainingData.tokenizer.specials.get("<pad>"));

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
//					this.network.clipGradNorm(1.0f);
					
					/**
					 * update
					 */
					this.network.update();

					/**
					 * current time error
					 */
					if(this.loss.isHasGPU()) {
						this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
					}else {
						this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
					}

					output.syncHost();

					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.tokenizer, trainingData.tokenizer.specials.get("<pad>"));

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
	
	public void trainNanoGPT_GEN(CNTokenizer trainingData) {
		// TODO Auto-generated method stub
		try {
			
			CUDAModules.initCUDAFunctions();

			this.dataSize = trainingData.number;
			
			if(isWarmUp()) {
				this.network.learnRate = (float) (this.lr * Math.pow(batchIndex * 1.0f/burnIn * 1.0f, power));
			}
			
			NanoGPT network = (NanoGPT) this.network;
			
			Tensor input = new Tensor(batchSize * network.time, 1, 1, 1, true);

			Tensor label = new Tensor(batchSize * network.time, 1, 1, network.vocabSize, true);
			
			Tensor mask = CNChatTokenizer.triu(batchSize, network.headNum, network.time, network.time, 1);

			Tensor positions = CNChatTokenizer.getPositions(batchSize, network.time);
			
			for(int i = 0;i<this.trainTime;i++) {
				
				if(this.trainIndex >= this.minTrainTime) {
					break;
				}
				
				this.trainIndex = i + 1;
				
				int[][] indexs = MathUtils.randomInts(trainingData.trainData.length - network.time,this.batchSize);
				
				Tensor output = null;
				
				/**
				 * 遍历整个训练集
				 */
				for(int it = 0;it<indexs.length;it++) {
					
					if(Math.abs(this.currentError) <= this.error) {
						break;
					}
					
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
					output = network.forward(input, positions, mask);
					
					/**
					 * loss
					 */
//					this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
					this.loss = network.loss(output, label);
					/**
					 * loss diff
					 */
//					this.lossDiff = network.lossDiff(output, label, trainingData.dictionary.get("<pad>"));
					this.lossDiff = network.lossDiff(output, label);
//					lossDiff.showDMByNumber(0);
					
//					System.out.println(JsonUtils.toJson(output.syncHost()));
					
//					GradClipping.gradClipping(this.lossDiff, 1e-7f);

					/**
					 * back
					 */
					this.network.back(this.lossDiff);
					
//					/**
//					 * grad clipping
//					 */
//					this.gradClipping(this.network);
					
//					this.network.clipGradNorm(1.0f);
					
					/**
					 * update
					 */
					this.network.update();

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
					
//					System.out.println(JsonUtils.toJson(inputEncoder.shape()));
//					System.out.println(JsonUtils.toJson(output.shape()));
//					System.out.println(JsonUtils.toJson(label.shape()));
					int time = output.number / batchSize;
					float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab);
					
//					if(error > 99) {
//						break;
//					}
					
					String msg = "training["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} train_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
					
					System.out.println(msg);

					this.batchIndex++;
					
					if(it != 0 && it % 200 == 0) {
						vail_gen(network, input, output, label, mask, positions, trainingData);
						network.RUN_MODEL = RunModel.TRAIN;
					}
				}
				
//				showOutputAndLabel(trainingData, inputEncoder, output, label, this.batchSize);

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
	
	@Override
	public void train(BaseData trainingData, BaseData testData) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void train(BaseData trainingData, BaseData validata, BaseData testData) {
		// TODO Auto-generated method stub
		
	}
	
	public void showOutputAndLabel(IndexDataLoader dataLoader,Tensor input,Tensor output,Tensor label,int batchSize) {
		String[] inputTxts = input2TXT(input, dataLoader, batchSize);
		String[] outputTxts = output2TXT(output, dataLoader, batchSize);
		String[] labelTxts = output2TXT(label, dataLoader, batchSize);
		for(int b = 0;b<batchSize;b++) {
			System.out.println("input:"+inputTxts[b]);
			System.out.println("output:"+outputTxts[b]);
			System.out.println("label :"+labelTxts[b]);
		}
	}
	
	public static String[] output2TXT(Tensor output,IndexDataLoader trainData,int batchSize) {
		String[] txts = new String[batchSize];
		for(int b = 0;b<batchSize;b++) {
			String txt = "";
			for(int t = 0;t<trainData.max_ch - 1;t++) {
				int charIndex = pickTopN(output.getByNumber(t * batchSize + b), 1);
				String c = trainData.ch_dic[charIndex];
				txt += c;
			}
			txts[b] = txt;
		}
		return txts;
	}
	
	public static String[] output2TXT(Tensor output,IndexDataLoader trainData,int time,int batchSize) {
		String[] txts = new String[batchSize];
		for(int b = 0;b<batchSize;b++) {
			String txt = "";
			for(int t = 0;t<time;t++) {
				int charIndex = pickTopN(output.getByNumber(t * batchSize + b), 1);
				String c = trainData.ch_dic[charIndex];
				txt += c;
			}
			txts[b] = txt;
		}
		return txts;
	}
	
	public static String[] input2TXT(Tensor input,IndexDataLoader trainData,int batchSize) {
		String[] txts = new String[batchSize];
		for(int b = 0;b<batchSize;b++) {
			String txt = "";
			for(int t = 0;t<trainData.max_en;t++) {
				int charIndex = pickTopN(input.getByNumber(t * batchSize + b), 1);
				String c = trainData.en_dic[charIndex];
				txt += c;
			}
			txts[b] = txt;
		}
		return txts;
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
	

	public String predict(IndexDataLoader trainingData,String input_txt) {
		
		Seq2Seq network = (Seq2Seq) this.network;
		
		Tensor input = trainingData.loadByTxt(input_txt);
		
		Tensor[] outputs = network.encoder(input);
		
		Tensor de_hx = outputs[1];
		Tensor de_cx = outputs[2];
		
		float[] data = new float[trainingData.ch_characters];
		data[trainingData.ch_dictionary.get("<BOS>")] = 1.0f;
		Tensor startInput = new Tensor(1, 1, 1, trainingData.ch_characters, data, true);
		
		String txt = "";
		
		for(int t = 0;t<network.de_time;t++) {
			
			Tensor output = network.decoder(de_hx, de_cx, startInput);
			output.syncHost();
			String[] txts = output2TXT(output, trainingData, 1, 1);
			
			txt += txts[0];
			
			startInput.clear();
			startInput.data[trainingData.ch_dictionary.get(txts[0])] = 1.0f;
			startInput.hostToDevice();
			
		}
		System.out.println(txt);
		return txt;
	}
	
	public String predictRNN(IndexDataLoader trainingData,String input_txt) {
		
		Seq2SeqRNN network = (Seq2SeqRNN) this.network;
		
		Tensor input = trainingData.loadByTxt(input_txt);
		
		Tensor en_hidden = network.encoder(input);
		en_hidden.syncHost();
		Tensor de_hidden = new Tensor(1, 1, 1, en_hidden.width, en_hidden.getByNumber(en_hidden.number - 1), true);
		
		float[] data = new float[trainingData.ch_characters];
		data[trainingData.ch_dictionary.get("<BOS>")] = 1.0f;
		Tensor startInput = new Tensor(1, 1, 1, trainingData.ch_characters, data, true);
		
		String txt = "";
		
		for(int t = 0;t<network.de_time;t++) {
			
			Tensor output = network.decoder(de_hidden, startInput);
			output.syncHost();
			String[] txts = output2TXT(output, trainingData, 1, 1);
			
			txt += txts[0];
			
			startInput.clear();
			startInput.data[trainingData.ch_dictionary.get(txts[0])] = 1.0f;
			startInput.hostToDevice();
			
		}
		System.out.println(txt);
		return txt;
	}
	
	public void vail_gen(NanoGPT network,Tensor input,Tensor output,Tensor label,Tensor mask,Tensor positions,CNTokenizer trainingData) {
		network.RUN_MODEL = RunModel.TEST;
		int[][] vailIndexs = MathUtils.randomInts(trainingData.vailData.length - network.time,this.batchSize);
		
		for(int it = 0;it<vailIndexs.length;it++) {
			
			if(it > 20) {
				break;
			}
			
			long start = System.nanoTime();
			
			this.loss.clear();

			/**
			 * 读取训练数据
			 */
			trainingData.loadDataVail(vailIndexs[it], input, label);
			/**
			 * forward
			 */
			output = network.forward(input, positions, mask);
			
			/**
			 * loss
			 */
			this.loss = network.loss(output, label);
			
			/**
			 * current time error
			 */
			if(this.loss.isHasGPU()) {
				this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
			}else {
				this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
			}

			output.syncHost();
			int time = output.number / batchSize;
			float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab);
			
			String msg = "vail["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} vail_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
			
			System.out.println(msg);
		}
		
	}
	
	public void vail_chat(NanoGPT network,Tensor input,Tensor output,Tensor label,Tensor positions,CNChatTokenizer trainingData) {
		network.RUN_MODEL = RunModel.TEST;
		int[][] vailIndexs = MathUtils.randomInts(trainingData.vailData.size() - network.time,this.batchSize);
		
		for(int it = 0;it<vailIndexs.length;it++) {
			
			if(it > 20) {
				break;
			}
			
			long start = System.nanoTime();
			
			this.loss.clear();

			/**
			 * 读取训练数据
			 */
			trainingData.loadVailData(vailIndexs[it], input, label);
			/**
			 * forward
			 */
			output = network.forward(input, positions);
			
			/**
			 * loss
			 */
			this.loss = network.loss(output, label, trainingData.dictionary.get("<pad>"));
			
			/**
			 * current time error
			 */
			if(this.loss.isHasGPU()) {
				this.currentError = MatrixOperation.sum(this.loss.syncHost()) / input.number;
			}else {
				this.currentError = MatrixOperation.sum(this.loss.data) / input.number;
			}

			output.syncHost();
			int time = output.number / batchSize;
			float error = this.accuracyBatchFisrt(input, output, label, time, batchSize, trainingData.vocab, trainingData.dictionary.get("<pad>"));
			
			String msg = "vail["+this.trainIndex+"]{"+it+"} (lr:"+this.network.learnRate+") accuracy:{"+error+"%} vail_loss:" + this.currentError + " [costTime:"+(System.nanoTime() - start)/1e6+"ms.]";
			
			System.out.println(msg);
		}
		
	}
	
}
