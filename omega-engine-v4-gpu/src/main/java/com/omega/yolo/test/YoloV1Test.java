package com.omega.yolo.test;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.data.DataSet;
import com.omega.engine.nn.layer.BasicBlockLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.pooling.PoolingType;
import com.omega.engine.updater.UpdaterType;
import com.omega.yolo.utils.LabelType;
import com.omega.yolo.utils.YoloDataLoader;

/**
 * 
 * @author Administrator
 *
 */
public class YoloV1Test {
	
	public static int im_w = 256;
	
	public static int im_h = 256;
	
	public float[] mean = new float[] {0.491f, 0.482f, 0.446f};
	public float[] std = new float[] {0.247f, 0.243f, 0.261f};
	
	public void showImg() {
		
		String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
		String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
		
		String outputPath = "H:\\voc\\sample_set\\";
		
		YoloDataLoader loader = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, false);
		
		Tensor imgSet = loader.getImgSet();
		
		Tensor labelSet = loader.getLabelSet();
		
		System.out.println("load data finish.");
		
		ImageUtils utils = new ImageUtils();
		
		for(int b = 0;b<imgSet.number;b++) {
			
			float[] once = imgSet.getByNumber(b);
			
			float[] label = labelSet.getByNumber(b);
			
			int[][] bbox = new int[][] {
					{	
						0,
						(int) label[1],
						(int) label[2],
						(int) label[3],
						(int) label[4]
					}
			};
			
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h), im_w, im_h, bbox);
			
		}
		
	}
	
	public static void showImg(String outputPath,DataSet dataSet,int class_num,float[][][] score_bbox,boolean format) {
		

		ImageUtils utils = new ImageUtils();
		
		
		for(int b = 0;b<dataSet.number;b++) {
			
			float[] once = dataSet.getOnceData(b).data;
			
			float[][] label = score_bbox[b];
			
			List<Integer> indexs = new ArrayList<Integer>();
			
			for(int l = 0;l<label.length;l++) {
				
				for(int c = 0;c<class_num;c++) {
					
					if(label[l][c] == 0.0f) {
						continue;
					}
					indexs.add(l);
				}
				
			}
			
			int[][] bbox = new int[indexs.size()][5];
			
			for(int i = 0;i<indexs.size();i++) {
				
				Integer index = indexs.get(i);
				
				bbox[i][0] = 0;
				bbox[i][1] = (int) (label[index][class_num + 1] - label[index][class_num + 3] / 2);
				bbox[i][2] = (int) (label[index][class_num + 2] - label[index][class_num + 4] / 2);
				bbox[i][3] = (int) (label[index][class_num + 1] + label[index][class_num + 3] / 2);
				bbox[i][4] = (int) (label[index][class_num + 2] + label[index][class_num + 4] / 2);
				
			}
			
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox);
			
		}
		
	}
	
	public int[][] tensorToRGB(Tensor t){
		int[][] rgb = new int[t.channel][t.height * t.width];
		
		for(int c = 0;c<t.channel;c++) {
			for(int i = 0;i<rgb.length;i++) {
				rgb[c][i] = (int)t.data[c * t.height * t.width + i];
			}
		}
		
		return rgb;
	}
	
	public void yolov1() {
		
		try {
			
			String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
			String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
			
			String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
			String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
			
			YoloDataLoader trainData = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, true);
			
			YoloDataLoader testData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, true);
			
			DataSet trainSet = formatToYolo(trainData.getDataSet());
			
			DataSet testSet = formatToYolo(testData.getDataSet());
			
			System.out.println("load data finish.");
			
			CNN netWork = new CNN(LossType.yolo3, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;
			
			int channel = 3;
			
			int height = 256;
			
			int width = 256;
			
			InputLayer inputLayer = new InputLayer(channel, height, width);
			
			ConvolutionLayer conv1 = new ConvolutionLayer(channel, 64, width, height, 7, 7, 3, 2, false);
			
			BNLayer bn1 = new BNLayer();
			
			LeakyReluLayer active1 = new LeakyReluLayer();
			
			PoolingLayer pool1 = new PoolingLayer(conv1.oChannel, conv1.oWidth, conv1.oHeight, 2, 2, 2, PoolingType.MAX_POOLING);
			
			/**
			 * block1  64 * 32 * 32
			 */
			BasicBlockLayer bl1 = new BasicBlockLayer(pool1.oChannel, 64, pool1.oHeight, pool1.oWidth, 1, netWork);
			LeakyReluLayer active2 = new LeakyReluLayer();

			/**
			 * block2  64 * 32 * 32
			 */
			BasicBlockLayer bl2 = new BasicBlockLayer(bl1.oChannel, 64, bl1.oHeight, bl1.oWidth, 1, netWork);
			LeakyReluLayer active3 = new LeakyReluLayer();
			
			/**
			 * block3  128 * 16 * 16
			 * downSample 32 / 2 = 16
			 */
			BasicBlockLayer bl3 = new BasicBlockLayer(bl2.oChannel, 128, bl2.oHeight, bl2.oWidth, 2, netWork);
			LeakyReluLayer active4 = new LeakyReluLayer();

			/**
			 * block4  128 * 16 * 16
			 */
			BasicBlockLayer bl4 = new BasicBlockLayer(bl3.oChannel, 128, bl3.oHeight, bl3.oWidth, 1, netWork);
			LeakyReluLayer active5 = new LeakyReluLayer();

			/**
			 * block5  256 * 8 * 8
			 * downSample 16 / 2 = 8
			 */
			BasicBlockLayer bl5 = new BasicBlockLayer(bl4.oChannel, 256, bl4.oHeight, bl4.oWidth, 2, netWork);
			LeakyReluLayer active6 = new LeakyReluLayer();
			
			/**
			 * block6  256 * 8 * 8
			 */
			BasicBlockLayer bl6 = new BasicBlockLayer(bl5.oChannel, 256, bl5.oHeight, bl5.oWidth, 1, netWork);
			LeakyReluLayer active7 = new LeakyReluLayer();

			/**
			 * block7  512 * 4 * 4
			 * downSample 8 / 2 = 4
			 */
			BasicBlockLayer bl7 = new BasicBlockLayer(bl6.oChannel, 512, bl6.oHeight, bl6.oWidth, 2, netWork);
			LeakyReluLayer active8 = new LeakyReluLayer();
			
			/**
			 * block8  512 * 4 * 4
			 */
			BasicBlockLayer bl8 = new BasicBlockLayer(bl7.oChannel, 512, bl7.oHeight, bl7.oWidth, 1, netWork);
			LeakyReluLayer active9 = new LeakyReluLayer();
			
			PoolingLayer pool2 = new PoolingLayer(bl8.oChannel, bl8.oWidth, bl8.oHeight, 3, 3, 2, PoolingType.MEAN_POOLING);
			
			ConvolutionLayer conv2 = new ConvolutionLayer(pool2.oChannel, 1024, pool2.oWidth, pool2.oHeight, 3, 3, 1, 1, false);
			BNLayer bn2 = new BNLayer();
			LeakyReluLayer active11 = new LeakyReluLayer();
			
			ConvolutionLayer conv3 = new ConvolutionLayer(conv2.oChannel, 256, conv2.oWidth, conv2.oHeight, 3, 3, 1, 1, false);
			BNLayer bn3 = new BNLayer();
			LeakyReluLayer active12 = new LeakyReluLayer();
			
			/**
			 * fully  512 * 7 * 7
			 */
			int fInputCount = conv3.oChannel * conv3.oWidth * conv3.oHeight;
			
			FullyLayer full1 = new FullyLayer(fInputCount, 539);
			SigmodLayer active13 = new SigmodLayer();
			
			netWork.addLayer(inputLayer);
			netWork.addLayer(conv1);
			netWork.addLayer(bn1);
			netWork.addLayer(active1);
			netWork.addLayer(pool1);
			
			/**
			 * block1  64
			 */
			netWork.addLayer(bl1);
			netWork.addLayer(active2);
			netWork.addLayer(bl2);
			netWork.addLayer(active3);
			
			/**
			 * block2  128
			 */
			netWork.addLayer(bl3);
			netWork.addLayer(active4);
			netWork.addLayer(bl4);
			netWork.addLayer(active5);
			
			/**
			 * block3  256
			 */
			netWork.addLayer(bl5);
			netWork.addLayer(active6);
			netWork.addLayer(bl6);
			netWork.addLayer(active7);
			
			/**
			 * block4  512
			 */
			netWork.addLayer(bl7);
			netWork.addLayer(active8);
			netWork.addLayer(bl8);
			netWork.addLayer(active9);
			
			netWork.addLayer(pool2);
			
			netWork.addLayer(conv2);
			netWork.addLayer(bn2);
			netWork.addLayer(active11);
			
			netWork.addLayer(conv3);
			netWork.addLayer(bn3);
			netWork.addLayer(active12);
			
			netWork.addLayer(full1);
			netWork.addLayer(active13);
			
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1000, 0.001f, 32, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.trainObjectRecognition(trainSet, testSet);
			
			
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
	
	public void yolov1_tiny() {
		
		try {
			
			String cfg_path = "H:/voc/train/yolov1-tiny.cfg";
			
			String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
			String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
			
			String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
			String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
			
			YoloDataLoader trainData = new YoloDataLoader(trainPath, trainLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, true);
			
			YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, true);
			
			DataSet trainSet = formatToYolo(trainData.getDataSet());
			
			DataSet vailSet = formatToYolo(vailData.getDataSet());
			
			System.out.println("load data finish.");
			
			CNN netWork = new CNN(LossType.yolo3, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;

			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 1000, 0.001f, 64, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.trainObjectRecognition(trainSet, vailSet);
			

			/**
			 * 处理测试预测结果
			 */
			float[][][] draw_bbox = optimizer.showObjectRecognition(vailSet, 64);
			
			YoloDataLoader testData = new YoloDataLoader(testPath, testLabelPath, 1000, 3, 256, 256, 5, LabelType.csv, false);
			
			String outputPath = "H:\\voc\\banana-detection\\test\\";
			
			showImg(outputPath, testData.getDataSet(), 1, draw_bbox, false);
			
			System.out.println(((System.currentTimeMillis() - start) / 1000) + "s.");
			
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

	public static DataSet formatToYolo(DataSet dataSet) {
		dataSet.label.data = labelToYolo(dataSet.label.data, dataSet.label.number, 7, im_w, im_h);
		dataSet.label.width = 7 * 7 * 6;
		dataSet.labelSize = 7 * 7 * 6;
		return dataSet;
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
	public static float[] labelToYolo(float[] bbox,int number,int stride,int im_w,int im_h) {
		
		float[] target = new float[number * stride * stride * 6];
		
		int oneSize = stride * stride * 6;

		for(int i = 0;i<number;i++) {
			
			float x1 = bbox[i * 5 + 1];
			float y1 = bbox[i * 5 + 2];
			float x2 = bbox[i * 5 + 3];
			float y2 = bbox[i * 5 + 4];
			
			float cx = (x1 + x2) / (2 * im_w);
			float cy = (y1 + y2) / (2 * im_h);
			
			float w = (x2 - x1) / im_w;
			float h = (y2 - y1) / im_h;

			int gridx = (int)(cx * stride);
			int gridy = (int)(cy * stride);
			
			float px = cx * stride - gridx;
			float py = cy * stride - gridy;
			
			/**
			 * c1
			 */
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 0] = 1.0f;
			
			/**
			 * class
			 */
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 1] = 1.0f;
			
			/**
			 * x1,y1,w1,h1
			 */
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 2] = px;
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 3] = py;
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 4] = w;
			target[i * oneSize + gridy * stride * 6 + gridx * 6 + 5] = h;
			
		}
		
		return target;
	}
	
	public static float[] yoloTolabel(float[] target,int number,int stride,int im_w,int im_h) {
		
		float[] bbox = new float[number * stride * stride * 5];
		
		int oneSize = 5;
		
		int oneTargetSize = stride * stride * 6;
		
		for(int i = 0;i<number;i++) {
			
			for(int l = 0;l<stride * stride;l++) {

				float c = target[i * oneTargetSize + l * 6];
				
				if(c <= 0.0f) {
					continue;
				}

				float px = target[i * oneTargetSize + l * 6 + 2];
				float py = target[i * oneTargetSize + l * 6 + 3];
				float w = target[i * oneTargetSize + l * 6 + 4];
				float h = target[i * oneTargetSize + l * 6 + 5];

				int row = l / stride;
		        int col = l % stride;
				
				float cx = (px + col) / stride;
	            float cy = (py + row) / stride;
	            
				bbox[i * oneSize + 0] = 1.0f;
				bbox[i * oneSize + 1] = (cx - w/2) * im_w;
				bbox[i * oneSize + 2] = (cy - h/2) * im_h;
				bbox[i * oneSize + 3] = (cx + w/2) * im_w;
				bbox[i * oneSize + 4] = (cy + h/2) * im_h;

			}
			
		}
		
		return bbox;
	}
	
	public static void showLabel(float[] bbox,int number,int stride,int im_w,int im_h) {
		
		float[] target = labelToYolo(bbox, number, stride, im_w, im_h);
		
		float[] label = yoloTolabel(target, number, stride, im_w, im_h);
		
		for(int i = 0;i<number;i++) {
			
			System.out.println("-------"+i+"--------");
			
			System.out.println(bbox[i * 5 + 0]+":"+label[i * 5 + 0]);
			System.out.println(bbox[i * 5 + 1]+":"+label[i * 5 + 1]);
			System.out.println(bbox[i * 5 + 2]+":"+label[i * 5 + 2]);
			System.out.println(bbox[i * 5 + 3]+":"+label[i * 5 + 3]);
			System.out.println(bbox[i * 5 + 4]+":"+label[i * 5 + 4]);
			
		}
		
	}
	
	public void testYoloBBox() {
		
		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
		
		YoloDataLoader vailData = new YoloDataLoader(testPath, testLabelPath, 100, 3, 256, 256, 5, LabelType.csv, true);
		
		showLabel(vailData.getLabelSet().data, 100, 7, im_w, im_h);
		
	}
	
	public static void main(String[] args) {
		
		CUDAModules.initContext();
		
		YoloV1Test t = new YoloV1Test();

		t.yolov1();
		
//		t.yolov1_tiny();
		
//		t.testYoloBBox();
		
//		t.showImg();

	}
	
}
