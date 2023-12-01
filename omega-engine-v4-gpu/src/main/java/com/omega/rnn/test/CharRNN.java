package com.omega.rnn.test;

import com.omega.engine.active.ActiveType;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.RNNBlockLayer;
import com.omega.engine.nn.layer.RNNLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.rnn.data.OneHotDataLoader;

public class CharRNN {
	
	public void charRNN() {
		
		try {
			
			int time = 576;
			
			int batchSize = 32;
			
			int hiddenSize = 512;
			
			String trainPath = "H:\\rnn_dataset\\shakespeare.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);

			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
			RNNLayer l1 = new RNNLayer(trainData.characters, hiddenSize, hiddenSize, time, ActiveType.leaky_relu, false, netWork);
			
			RNNLayer l2 = new RNNLayer(hiddenSize, hiddenSize, hiddenSize, time, ActiveType.leaky_relu, false, netWork);
			
			RNNLayer l3 = new RNNLayer(hiddenSize, hiddenSize, hiddenSize, time, ActiveType.leaky_relu, false, netWork);
			
			FullyLayer f1 = new FullyLayer(hiddenSize, trainData.characters, true);
			LeakyReluLayer a1 = new LeakyReluLayer();
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(l1);
			netWork.addLayer(l2);
			netWork.addLayer(l3);
			netWork.addLayer(f1);
			netWork.addLayer(a1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.1f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.POLY, false);

//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void charRNN2() {
		
		try {
			
			int time = 128;
			
			int batchSize = 512;
			
			int hiddenSize = 256;
			
			int rnnLayerNum = 2;
			
			float dropout = 0.5f;
			
			boolean bidirectional = false;
			
			int rnnMode = 2;
			
			String trainPath = "H:\\rnn_dataset\\shakespeare.txt";
			
			OneHotDataLoader trainData = new OneHotDataLoader(trainPath, time, batchSize);

			RNN netWork = new RNN(LossType.softmax_with_cross_entropy, UpdaterType.adamw, time);
			
			InputLayer inputLayer = new InputLayer(1, 1, trainData.characters);
			
			RNNBlockLayer l1 = new RNNBlockLayer(time, rnnLayerNum, trainData.characters, hiddenSize, rnnMode, bidirectional, dropout, netWork);
			
			FullyLayer f1 = new FullyLayer(hiddenSize, trainData.characters, true);
			LeakyReluLayer a1 = new LeakyReluLayer();
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(l1);
			netWork.addLayer(f1);
			netWork.addLayer(a1);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 500, 0.001f, batchSize, LearnRateUpdate.POLY, false);

//			long start = System.currentTimeMillis();
			
			optimizer.trainRNN(trainData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			CharRNN t = new CharRNN();
			
			t.charRNN2();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
			
		}
		
	}
	
}
