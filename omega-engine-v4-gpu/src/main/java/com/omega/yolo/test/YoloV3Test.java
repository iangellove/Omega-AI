package com.omega.yolo.test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.DarknetLoader;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.network.Yolo;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.yolo.data.DataType;
import com.omega.yolo.data.YoloDataTransform2;
import com.omega.yolo.model.YoloBox;
import com.omega.yolo.model.YoloDetection;
import com.omega.yolo.utils.AnchorBoxUtils;
import com.omega.yolo.utils.DetectionDataLoader;
import com.omega.yolo.utils.LabelFileType;
import com.omega.yolo.utils.LabelType;
import com.omega.yolo.utils.LabelUtils;
import com.omega.yolo.utils.YoloDataLoader;
import com.omega.yolo.utils.YoloLabelUtils;

public class YoloV3Test {
	
	
	public static void showImg(String outputPath,DetectionDataLoader dataSet,int class_num,List<YoloBox> score_bbox,int batchSize,boolean format,int im_w,int im_h) {
		

		ImageUtils utils = new ImageUtils();
		
		int lastIndex = dataSet.number % batchSize;
		
		for(int b = 0;b<dataSet.number;b++) {
			
			float[] once = dataSet.loadData(b);
			
			int bbox_index = b;
			
			if(b >= (dataSet.number - lastIndex)) {
				bbox_index = b + (batchSize - lastIndex);
			}
			
			YoloBox box = score_bbox.get(bbox_index);

			List<Integer> indexs = new ArrayList<Integer>();
			
			for(int l = 0;l<box.getDets().size();l++) {

				if(box.getDets().get(l) != null && box.getDets().get(l).getObjectness() > 0) {
					indexs.add(l);
				}
				
			}
			
			int[][] bbox = new int[indexs.size()][5];
			
			for(int i = 0;i<indexs.size();i++) {
				
				Integer index = indexs.get(i);

				YoloDetection det = box.getDets().get(index);
				
				bbox[i][0] = 0;
				bbox[i][1] = (int) ((det.getBbox()[0] - det.getBbox()[2] / 2.0f) * im_w);
				bbox[i][2] = (int) ((det.getBbox()[1] - det.getBbox()[3] / 2.0f) * im_h);
				bbox[i][3] = (int) ((det.getBbox()[0] + det.getBbox()[2] / 2.0f) * im_w);
				bbox[i][4] = (int) ((det.getBbox()[1] + det.getBbox()[3] / 2.0f) * im_h);
				
			}
			
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, im_w, im_h, format), im_w, im_h, bbox);
			
		}
		
	}
	
	public void yolov3_tiny() {
		
		int im_w = 256;
		int im_h = 256;
		int batchSize = 64;
		int class_num = 1;
		
		try {
			
			String cfg_path = "H:\\voc\\banana-detection\\yolov3-tiny-banana.cfg";
			
			String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
			String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
			
			String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
			String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
			
			YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
			
			DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
			
			DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.csv, im_w, im_h, class_num, batchSize, DataType.yolov3);

			Yolo netWork = new Yolo(LossType.yolo3, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;

			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 2000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);
			
			optimizer.lr_step = new int[] {200,500,1000,1200,2000};
			
			optimizer.trainObjectRecognitionOutputs(trainData, vailData);
			
			/**
			 * 处理测试预测结果
			 */
			List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
			
			String outputPath = "H:\\voc\\banana-detection\\test_yolov3\\";
			
			showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h);
		
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
			
	}
	
	public void yolov3_tiny_helmet() {
		
		int im_w = 416;
		int im_h = 416;
		int batchSize = 24;
		int class_num = 5;
		
		try {
			
			String cfg_path = "H:\\voc\\helmet_dataset\\yolov3-tiny-helmet.cfg";
			
			String trainPath = "H:\\voc\\helmet_dataset\\train";
			String trainLabelPath = "H:\\voc\\helmet_dataset\\train_label.txt";
			
			String testPath = "H:\\voc\\helmet_dataset\\vail";
			String testLabelPath = "H:\\voc\\helmet_dataset\\vail_label.txt";
			
			YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
			
			DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
			
			DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);

			Yolo netWork = new Yolo(LossType.yolo3, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;

			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 5000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);

			optimizer.trainObjectRecognitionOutputs(trainData, vailData);
			
//			/**
//			 * 处理测试预测结果
//			 */
//			List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
//			
//			String outputPath = "H:\\voc\\helmet_dataset\\test_yolov3\\";
//			
//			showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h);
		
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
			
	}
	
	public void yolov3_tiny_voc() {
		
		int im_w = 416;
		int im_h = 416;
		int batchSize = 24;
		int class_num = 20;
		
		try {
			
			String cfg_path = "H:/voc/train/yolov3-tiny-voc.cfg";
			
			String trainPath = "H:\\voc\\train\\imgs";
			String trainLabelPath = "H:\\voc\\train\\labels\\yolov3.txt";
			
			String testPath = "H:\\voc\\test\\imgs";
			String testLabelPath = "H:\\voc\\test\\labels\\yolov3.txt";
			
			String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.conv.15";
			
//			String weightPath = "H:\\voc\\yolo-weights\\yolov3-tiny.weights";
			
			YoloDataTransform2 dt = new YoloDataTransform2(class_num, DataType.yolov3, 90);
			
			DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3, dt);
			
			DetectionDataLoader vailData = new DetectionDataLoader(testPath, testLabelPath, LabelFileType.txt, im_w, im_h, class_num, batchSize, DataType.yolov3);

			Yolo netWork = new Yolo(LossType.yolo3, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.001f;

			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			DarknetLoader.loadWeight(netWork, weightPath, 14, true);
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 3000, 0.001f, batchSize, LearnRateUpdate.SMART_HALF, false);

//			optimizer.lr_step = new int[] {200,500,1000,1500,2000};
			
			optimizer.trainObjectRecognitionOutputs(trainData, vailData);
			
//			
//			/**
//			 * 处理测试预测结果
//			 */
//			List<YoloBox> draw_bbox = optimizer.showObjectRecognitionYoloV3(vailData, batchSize);
//			
//			String outputPath = "H:\\voc\\banana-detection\\test_yolov3\\";
//			
//			showImg(outputPath, vailData, class_num, draw_bbox, batchSize, false, im_w, im_h);

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
			
	}
	
	public void getAnchors() {
		
		try {

			String trainLabelPath = "H:\\voc\\helmet_dataset\\train_label.txt";
			
			Tensor bbox = LabelUtils.loadBoxTXT(trainLabelPath);
			
			Tensor anchors = AnchorBoxUtils.getAnchorBox(bbox, 6);
			
			System.out.println(JsonUtils.toJson(anchors.data));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void yolov3_show() {
		
		int im_w = 416;
		int im_h = 416;
		int classNum = 20;
		int batchSize = 64;
		
		String trainPath = "H:\\voc\\train\\imgs";
		String trainLabelPath = "H:\\voc\\train\\labels\\yolov3.txt";
		
		String outputPath = "H:\\voc\\train\\show\\";
		
//		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
//		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
		
		YoloDataTransform2 dt = new YoloDataTransform2(classNum, DataType.yolov3, 90);
		
		DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3, dt);
		
		Tensor input = new Tensor(batchSize, 3, im_w, im_h);
		
		Tensor label = trainData.initLabelTensor();
		
		for(int i = 0;i<1;i++) {

			int[] indexs = MatrixUtils.orderInt(batchSize, i);
			
			trainData.showImg(outputPath, indexs, input, label);
			
		}
		
	}
	
	public void yolov3_show2() {
		
		int im_w = 256;
		int im_h = 256;
		int classNum = 1;
		int batchSize = 64;
		
		String trainPath = "H:\\voc\\banana-detection\\bananas_train\\images";
		String trainLabelPath = "H:\\voc\\banana-detection\\bananas_train\\label.csv";
		
		String outputPath = "H:\\voc\\banana-detection\\bananas_train\\show\\";
		
//		String testPath = "H:\\voc\\banana-detection\\bananas_val\\images";
//		String testLabelPath = "H:\\voc\\banana-detection\\bananas_val\\label.csv";
		
		YoloDataTransform2 dt = new YoloDataTransform2(classNum, DataType.yolov3, 90);
		
		DetectionDataLoader trainData = new DetectionDataLoader(trainPath, trainLabelPath, LabelFileType.csv, im_w, im_h, classNum, batchSize, DataType.yolov3, dt);
		
		Tensor input = new Tensor(batchSize, 3, im_w, im_h);
		
		Tensor label = trainData.initLabelTensor();
		
		for(int i = 0;i<1;i++) {

			int[] indexs = MatrixUtils.orderInt(batchSize, i);
			
			trainData.showImg(outputPath, indexs, input, label);
			
		}
		
	}
	
	public static void createTrainTestDataSet() {
		
		try {

			int im_w = 416;
			int im_h = 416;
			int classNum = 5;
			int batchSize = 64;
			float trainRatio = 0.8f;
			
			String orgPath = "H:\\voc\\helmet_dataset\\imgs\\";
			String orgLabelPath = "H:\\voc\\helmet_dataset\\labels\\yolov3.txt";
			
			String trainPath = "H:\\voc\\helmet_dataset\\train\\";
			String vailPath = "H:\\voc\\helmet_dataset\\vail\\";
			
			String trainLabelPath = "H:\\voc\\helmet_dataset\\train_label.txt";
			String vailLabelPath = "H:\\voc\\helmet_dataset\\vail_label.txt";
			
			DetectionDataLoader orgData = new DetectionDataLoader(orgPath, orgLabelPath, LabelFileType.txt, im_w, im_h, classNum, batchSize, DataType.yolov3);
			
			List<Integer> list = new ArrayList<Integer>(); 
			
			for(int i = 0;i<orgData.idxSet.length;i++) {
				list.add(i);
			}
			
			Collections.shuffle(list);
			
			System.out.println(JsonUtils.toJson(list));
			
			int trainSize = Math.round(orgData.idxSet.length * trainRatio);
			int testSize = orgData.idxSet.length - trainSize;
			
			Map<String,float[]> trainLabelData = new HashMap<String, float[]>();
			Map<String,float[]> testLabelData = new HashMap<String, float[]>();
			
			/**
			 * 复制文件
			 */
			for(int b = 0;b<trainSize;b++) {
				int index = list.get(b);
				String filename = orgData.idxSet[index];
				if(orgData.orgLabelData.get(filename).length <= 450) {
					File file = new File(orgPath+filename+".jpg");
					File outFile = new File(trainPath+filename+".jpg");
					copyFileUsingStream(file, outFile);
					trainLabelData.put(filename, orgData.orgLabelData.get(filename));
				}
			}
			
			for(int b = 0;b<testSize;b++) {
				int index = list.get(b+trainSize);
				String filename = orgData.idxSet[index];
				if(orgData.orgLabelData.get(filename).length <= 450) {
					File file = new File(orgPath+filename+".jpg");
					File outFile = new File(vailPath+filename+".jpg");
					copyFileUsingStream(file, outFile);
					testLabelData.put(filename, orgData.orgLabelData.get(filename));
				}
			}
			
			/**
			 * 复制label
			 */
			createLabelTXT(trainLabelPath, trainLabelData);
			
			createLabelTXT(vailLabelPath, testLabelData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	

	public static void copyFileUsingStream(File source, File dest) throws IOException {
	    InputStream is = null;
	    OutputStream os = null;
	    try {
	        is = new FileInputStream(source);
	        os = new FileOutputStream(dest);
	        byte[] buffer = new byte[1024];
	        int length;
	        while ((length = is.read(buffer)) > 0) {
	            os.write(buffer, 0, length);
	        }
	    } finally {
	        is.close();
	        os.close();
	    }
	}
	
	public static void createLabelTXT(String txtPath,Map<String,float[]> data) {
		File txt = new File(txtPath);
		
		if(!txt.exists()) {
			try {
				txt.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} // 创建新文件,有同名的文件的话直接覆盖
		}
		
		try (FileOutputStream fos = new FileOutputStream(txt);) {
 
			for (String name : data.keySet()) {
				
				String text = name;
				
				for(float val:data.get(name)) {
					text += " " + Math.round(val);
				}
				text += "\n";
				fos.write(text.getBytes());
			}
 
			fos.flush();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		
		try {
			
			CUDAModules.initContext();
			
			YoloV3Test y = new YoloV3Test();
			y.yolov3_tiny_voc();
//			y.yolov3_tiny();
//			y.getAnchors();
//			y.yolov3_show();
//			y.createTrainTestDataSet();
//			y.yolov3_tiny_helmet();
//			y.yolov3_show();
//			y.yolov3_show2();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
