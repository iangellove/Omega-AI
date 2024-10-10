package com.omega.example.clip.test;

import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.yolo.data.ImageLoader;

public class CLIPTest {
	
	public static void loadWeight(Map<String, Object> weightMap, ClipVision network) {
		for(String key:weightMap.keySet()) {
			System.out.println(key);
		}
		
		
	}
	
	public static void clip_test() {
		
		String clipWeight = "H:\\model\\clip_vision_weights.json";
		
		boolean bias = true;
		
		int channel = 3;
		int imgSize = 224;
		int patchSize = 32;
		
		int headNum = 12;
		int nLayers = 12;
		int clip_time = 50;
		int embedDim = 768;
		
		ClipVision network = new ClipVision(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, channel, imgSize, patchSize, headNum, nLayers, clip_time, embedDim, bias, false);
		
		loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network);
		
	}
	
	public static void imageProcessor() {
		
		String imgPath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\GCC_train_000000041.jpg";
		
		int w = 224;
		int h = 224;
		
		float[] mean = new float[] {0.48145466f, 0.4578275f, 0.40821073f};
		float[] std = new float[] {0.26862954f, 0.26130258f, 0.27577711f};
		
		Tensor input = ImageLoader.loadImage(imgPath, w, h, mean, std);
		
        input.showDM();
        
        /**
		 * print image
		 */
        MBSGDOptimizer.showImgs("H:\\testImg\\", input, "0", mean, std);
        
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			clip_test();
			
//			imageProcessor();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
