package com.omega.example.duffsion.test;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.DuffsionUNet;
import com.omega.engine.nn.network.SimpleUNet;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.duffsion.utils.DuffsionImageDataLoader;
import com.omega.example.unet.utils.SegImageDataLoader;

public class DuffsionModelTest {
	
	
	public static void duffsion_anime() {
		
		try {
			
			boolean bias = false;
			
			int batchSize = 4;
			int imw = 96;
			int imh = 96;
			int mChannel = 64;
			int resBlockNum = 2;
			int T = 1000;
			
			int[] channelMult = new int[] {1, 2, 3, 4};
			
			String imgDirPath = "H://voc//gan_anime//faces//";
			
			DuffsionImageDataLoader dataLoader = new DuffsionImageDataLoader(imgDirPath, imw, imh, batchSize);
			
			DuffsionUNet network = new DuffsionUNet(LossType.MSE, UpdaterType.adamw, T, 3, mChannel, channelMult, resBlockNum, imw, imh, bias);
			network.CUDNN = true;
			network.learnRate = 0.001f;
			
			MBSGDOptimizer optimizer = new MBSGDOptimizer(network, 50, 0.00001f, batchSize, LearnRateUpdate.GD_GECAY, false);
			
			optimizer.trainGaussianDiffusion(dataLoader);
			
			
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

	}
	
	public static void testCell(String path,String labelPath,String filename,String extName,SegImageDataLoader dataLoader,int height,int width,SimpleUNet network,Tensor input,Tensor label) {

		String testImg = path + filename + "." + extName;
		
		String testLabelImg = labelPath + filename + "." + extName;
		System.out.println(testImg);
		dataLoader.loadData(testImg, input);
		
		dataLoader.loadLabelData(testLabelImg, label);
		
		Tensor output = network.forward(input);
		
		ImageUtils utils = new ImageUtils();
		
		output.syncHost();
//	    PrintUtils.printImage(output);
	    for(int i = 0;i<output.dataLength;i++) {
	    	output.data[i] = output.data[i] * 255.0f;
		}

		String outImg = "H:\\unet-dataset\\cell\\" + filename + "." + extName;

		utils.createRGBImage(outImg, "png", ImageUtils.color2rgb2(output.data, output.channel, output.height, output.width, false), output.height, output.width, null, null);
		
		label.syncHost();
		for(int i = 0;i<label.dataLength;i++) {
			label.data[i] = label.data[i] * 255.0f;
		}

		String labelImg = "H:\\unet-dataset\\cell\\" + filename + "-mask" + "." + extName;
		
		utils.createRGBImage(labelImg, "png", ImageUtils.color2rgb2(label.data, label.channel, label.height, label.width, false), label.height, label.width, null, null);

	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			duffsion_anime();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
}
