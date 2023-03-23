package com.omega.yolo.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.model.ModelLoader;
import com.omega.engine.nn.network.CNN;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.yolo.utils.YoloDataLoader;

/**
 * 
 * @author Administrator
 *
 */
public class YoloV1Test {
	
	public void yolov1() {
		
		try {
			
			String imgDirPath = "H:/voc/train/imgs";
			String labelPath = "H:/voc/train/labels/bbox.txt";
			
			String cfg_path = "H:/voc/train/yolov1.cfg";
			
			YoloDataLoader loader = new YoloDataLoader(imgDirPath, labelPath, 4, 3, 448, 448, 1470);
	    	
//			DataSet testData = DataLoader.getImagesToDataSetByBin(test_data_filename, 10000, 3, 32, 32, 10, labelSet, true, mean, std);
			
			System.out.println("data is ready.");

			CNN netWork = new CNN(LossType.yolo, UpdaterType.adamw);
			
			netWork.CUDNN = true;
			
			netWork.learnRate = 0.01f;
			
			ModelLoader.loadConfigToModel(netWork, cfg_path);
			
			System.out.println("model conifg is ready.");
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(netWork, 200, 0.0001f, 4, LearnRateUpdate.CONSTANT, false);

			long start = System.currentTimeMillis();
			
			optimizer.train(loader);

//			optimizer.test(testData);
			
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
	
	public static void main(String[] args) {
		
		YoloV1Test t = new YoloV1Test();

    	CUDAModules.initContext();
		
		t.yolov1();
		
	}
	
}
