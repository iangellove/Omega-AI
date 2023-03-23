package com.omega.yolo.test;

import java.math.BigDecimal;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatlabDataLoader;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.loss.SoftmaxWithCrossEntropyLoss;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;

/**
 * 
 * @author Administrator
 *
 */
public class YoloV1Test2 {
	
	public void yolov1() {
		
		try {
			
			String trainPath = "H:\\voc\\training_set.mat";
			String testPath = "H:\\voc\\validation_set.mat";
			
			DataSet trainSet = MatlabDataLoader.loadMatData(trainPath);
			DataSet testSet = MatlabDataLoader.loadMatData(testPath);
			
			testSet.label = formatToYolo(testSet.label);
			
			System.out.println("load data finish.");
			
			CNN netWork = new CNN(LossType.yolo, UpdaterType.adam);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;
			
			int channel = 3;
			
			int height = 256;
			
			int width = 256;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 7, 7, 3, 2, false); //128 * 128
			
			BNLayer bn1 = new BNLayer();
			
			ReluLayer active1 = new ReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING); //64 * 64
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool1.oChannel, 64, pool1.width, pool1.height, 3, 3, 1, 2, false); //32
			
			BNLayer bn2 = new BNLayer();
			
			ReluLayer active2 = new ReluLayer();
			
			ConvolutionLayer conv3 = new ConvolutionLayer(conv2.oChannel, 128, conv2.width, conv2.height, 3, 3, 1, 2, false);  //16
			
			BNLayer bn3 = new BNLayer();
			
			ReluLayer active3 = new ReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(conv3.oChannel, conv3.oWidth, conv3.oHeight, 2, 2, 2, PoolingType.MAX_POOLING); //8 * 8
			
			/**
			 * fully
			 */
			int fInputCount = pool2.oChannel * pool2.oWidth * pool2.oHeight;

			int inputCount = 4096;
			
			FullyLayer full1 = new FullyLayer(fInputCount, inputCount, false);
			
//			BNLayer bn4 = new BNLayer();
			
			ReluLayer active4 = new ReluLayer();
			
			FullyLayer full2 = new FullyLayer(inputCount, 539, true);
			
			
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active2);
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(active3);
			netWork.addLayer(pool2);
			netWork.addLayer(full1);
//			netWork.addLayer(bn4);
			netWork.addLayer(active4);
			netWork.addLayer(full2);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 100, 0.001f, 4, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(trainSet);
			
			
//			System.out.println(JsonUtils.toJson(trainSet.label.data));
			
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}finally {
			try {
				CUDAMemoryManager.freeAll();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
	}
	
	public static Tensor formatToYolo(Tensor label) {
		label.data = labelToYolo(label.data, label.number, 7);
		label.width = 7 * 7 * 11;
		return label;
	}
	
	/**
	 * labelToLocation
	 * @param cx
	 * @param cy
	 * @param w
	 * @param h
	 * @param cla = 20
	 * @param stride = 7
	 * @param wimg = 448
	 * @param himg = 448
	 * @return 7 * 7 * (2 + 8 + 20) = 1470
	 * target = [px1,py1,w1,h1,c1,px2,py2,w2,h2,c2,clazz1.....,clazz20]
	 * gridx = int(cx / stride)
	 * gridy = int(cy / stride)
	 * px = (cx - (gridx * cellSize)) / cellSize
	 * py = (cy - (gridy * cellSize)) / cellSize
	 */
	public static float[] labelToYolo(float[] bbox,int number,int stride) {
		
		float cellSize = 1.0f / stride;
		
		float[] target = new float[stride * stride * 11];

		for(int i = 0;i<number;i++) {
			
			float cx = bbox[i * 5 + 1];
			float cy = bbox[i * 5 + 2];
			float w = bbox[i * 5 + 3];
			float h = bbox[i * 5 + 4];

			int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue();
			int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue();
			
//			System.out.println(cx+":"+cy+":"+gridx+":"+gridy);
			
			/**
			 * c1
			 */
			target[gridx * stride * 11 + gridy * 11 + 0] = 1.0f;
			/**
			 * c2
			 */
			target[gridx * stride * 11 + gridy * 11 + 5] = 1.0f;
			
			float px = (cx - gridx * cellSize) / cellSize;
			float py = (cy - gridy * cellSize) / cellSize;
			
			/**
			 * x1,y1,w1,h1
			 */
			target[gridx * stride * 11 + gridy * 11 + 1] = px;
			target[gridx * stride * 11 + gridy * 11 + 2] = py;
			target[gridx * stride * 11 + gridy * 11 + 3] = w;
			target[gridx * stride * 11 + gridy * 11 + 4] = h;
			/**
			 * x2,y2,w2,h2
			 */
			target[gridx * stride * 11 + gridy * 11 + 6] = px;
			target[gridx * stride * 11 + gridy * 11 + 7] = py;
			target[gridx * stride * 11 + gridy * 11 + 8] = w;
			target[gridx * stride * 11 + gridy * 11 + 9] = h;
		
			/**
			 * class
			 */
			target[gridx * stride * 11 + gridy * 11 + 10] = 1.0f;

		}
		
		return target;
	}
	
	public static void main(String[] args) {
		
		CUDAModules.initContext();
		
		YoloV1Test2 t = new YoloV1Test2();

		t.yolov1();
		
	}
	
}
