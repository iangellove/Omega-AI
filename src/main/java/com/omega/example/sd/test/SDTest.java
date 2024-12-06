package com.omega.example.sd.test;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.transformer.utils.ModelUtils;

import jcuda.driver.JCudaDriver;

/**
 * stable diffusion
 * @author Administrator
 *
 */
public class SDTest {
	
	public static void test_vqvae() {

		int batchSize = 1;
		int imageSize = 256;
		int z_dims = 64;
		int latendDim = 4;
		
		int num_vq_embeddings = 512;
		
		int num_res_blocks = 2;
		
		int[] channels = new int[] {64, 128, 256};
		boolean[] attn_resolutions = new boolean[] {false, false, false};
		
		float[] mean = new float[] {0.5f, 0.5f,0.5f};
		float[] std = new float[] {0.5f, 0.5f,0.5f};
		
		String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
		
		DiffusionImageDataLoader dataLoader = new DiffusionImageDataLoader(imgDirPath, imageSize, imageSize, batchSize, true, false, mean, std);
		
		TinyVQVAE2 network = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
		network.CUDNN = true;
		network.learnRate = 0.001f;
		network.RUN_MODEL = RunModel.TEST;
		
		String vqvae_model_path = "H:\\model\\vqvae2_500.model";
		ModelUtils.loadModel(network, vqvae_model_path);
		
		int[] indexs = new int[] {0};
		
		Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
		
		dataLoader.loadData(indexs, input);
		
		JCudaDriver.cuCtxSynchronize();
		
		Tensor out = network.forward(input);
		
//		Tensor latent = network.encode(input);
//		
//		latent.showShape();
//		
//		latent.showDM();
//		
//		Tensor out = network.decode(latent);
		
		out.showShape();
//		out.showDM();
		
		out.syncHost();
		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
		
		/**
		 * print image
		 */
		MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test", mean, std);
		
	}
	
	public static void sd_pokem() {
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
//			sd_pokem();
			
			test_vqvae();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
