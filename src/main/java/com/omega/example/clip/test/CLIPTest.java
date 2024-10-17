package com.omega.example.clip.test;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.yolo.data.ImageLoader;

public class CLIPTest {
	
	public static void replace_test() {
		
		BaseKernel kenel = new BaseKernel();
		
		int B = 16;
		int C = 512;
		int W = 512;
		int dataLen = B * C * W;
		
		int C2 = 50;
		
		float[] data = MatrixUtils.order(dataLen, 0);
		
		Tensor x1 = new Tensor(B, C, 1, W, data, true);
		
		float[] data2 = MatrixUtils.order(B * C2 * W, 0.1f, 0.01f);
		
		Tensor x2 = new Tensor(B, C, 1, W, data2, true);
		
		Tensor output = new Tensor(B, C, 1, W, true);
		
		float[] indices_data = new float[] {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8};
		
		Tensor indices = new Tensor(B, 1, 1, 1, indices_data, true);
		
		kenel.replace_channel_forward(x1, x2, output, indices, C2, B, C, 1, W);
		
//		x1.showDM();
//		
//		x2.showDM();
		
		output.showDM();
		
//		Tensor dx = new Tensor(2, 2, 1, 10, true);
//		
//		float[] diff_data = MatrixUtils.order(100, 0.01f, 0.01f);
//		
//		Tensor diff = new Tensor(2, 5, 1, 10, diff_data, true);
//		
//		kenel.replace_channel_backward(diff, dx, indices, 2);
//		
//		diff.showDM();
//		
//		dx.showDM();
	}
	
	public static void clip_test() {
		
		boolean bias = true;
		
		int channel = 3;
		int imgSize = 224;
		int patchSize = 32;
		
		int headNum = 12;
		int nLayers = 12;
		int clip_time = 50;
		int embedDim = 768;
		
		ClipVision network = new ClipVision(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, channel, imgSize, patchSize, headNum, nLayers, clip_time, embedDim, bias, false);
		network.time = 50;
		network.CUDNN = true;
		
		String clipWeight = "H:\\model\\clip_vision_weights.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, true);
		
		Tensor input = new Tensor(4, 3, imgSize, imgSize, true);
		
		imageProcessor(input);
		
		network.forward(input);
		
		network.getEncoder().getImageEncoders().showShape();
		network.getEncoder().getImageEncoders().showDM();
//		network.getEncoder().getImageEncoders().showDMByOffset(0, 50 * 768);
//		network.getEncoder().getImageEncoders().showDMByOffset(50 * 768, 50 * 768);
	}
	
	public static Tensor imageProcessor(Tensor input) {
		
//		String imgPath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\GCC_train_000000041.jpg";
		
		String imgPath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\GCC_train_002582585.jpg";
		String imgPath2 = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\GCC_train_002582585.jpg";
		
		int w = 224;
		int h = 224;
		
		float[] mean = new float[] {0.48145466f, 0.4578275f, 0.40821073f};
		float[] std = new float[] {0.26862954f, 0.26130258f, 0.27577711f};
		
//		float[] mean = new float[] {0f, 0f, 0f};
//		float[] std = new float[] {1f, 1f, 1f};
		
		ImageLoader.loadImage(input, 0, imgPath, w, h, mean, std, true);
		ImageLoader.loadImage(input, 1, imgPath2, w, h, mean, std, true);
		ImageLoader.loadImage(input, 2, imgPath, w, h, mean, std, true);
		ImageLoader.loadImage(input, 3, imgPath, w, h, mean, std, true);
		input.hostToDevice();
//		input.showDMByNumber(0);
//		input.showDMByNumber(1);
//		PrintUtils.printImage(input);
//        input.showDM();
        
//        /**
//		 * print image
//		 */
//        MBSGDOptimizer.showImgs("H:\\testImg\\", input, "0", mean, std);
        
        return input;
        
	}

	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			clip_test();
			
//			replace_test();
			
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
