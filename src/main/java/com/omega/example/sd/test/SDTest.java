package com.omega.example.sd.test;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipText;
import com.omega.engine.nn.network.DiffusionUNetCond;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.nn.network.vae.TinyVQVAE2;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.diffusion.utils.DiffusionImageDataLoader;
import com.omega.example.sd.utils.SDImageDataLoader;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;

import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;

/**
 * stable diffusion
 * @author Administrator
 *
 */
public class SDTest {
	
	public static void test_vqvae() {

		int batchSize = 2;
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
		network.RUN_MODEL = RunModel.EVAL;
		
		String vqvae_model_path = "H:\\model\\vqvae2_256_500.model";
		ModelUtils.loadModel(network, vqvae_model_path);
		
		int[] indexs = new int[] {0, 1, 2, 3};
		
		Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
		
		dataLoader.loadData(indexs, input);
		
		JCudaDriver.cuCtxSynchronize();
		
//		Tensor out = network.forward(input);
		
		Tensor latent = network.encode(input);
		
//		latent.showShape();
		
//		latent.showDM();
		
		Tensor out = network.decode(latent);
		
		out.showShape();
//		out.showDM();
		
		out.syncHost();
		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
		
		/**
		 * print image
		 */
		MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test", mean, std);
		
		indexs = new int[] {4, 5, 6, 7};
		
		dataLoader.loadData(indexs, input);
		
		JCudaDriver.cuCtxSynchronize();
		
//		Tensor out = network.forward(input);
		for(int i = 0;i<10;i++) {
			long start = System.nanoTime();
			latent = network.encode(input);
			out = network.decode(latent);
			JCuda.cudaDeviceSynchronize();
			System.err.println((System.nanoTime() - start)/1e6+"ms.");
		}
		
		
		out.showShape();
//		out.showDM();
		
		out.syncHost();
		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
		
		/**
		 * print image
		 */
		MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test1", mean, std);
		
	}
	
	public static void test_vqvae32() {

		int batchSize = 2;
		int imageSize = 256;
		int z_dims = 32;
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
		network.RUN_MODEL = RunModel.EVAL;
		
		String vqvae_model_path = "H:\\model\\vqvae2_32_256_500.model";
		ModelUtils.loadModel(network, vqvae_model_path);
		
		String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
		
		String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";

		boolean horizontalFilp = true;
		
		int imgSize = 256;
		
		int maxContextLen = 64;
		
		SDImageDataLoader dataLoader2 = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
		
		int time = maxContextLen;
		int maxPositionEmbeddingsSize = 512;
		int vocabSize = 21128;
		int hiddenSize = 768;
		int typeVocabSize = 2;
		int headNum = 12;
		int numHiddenLayers = 12;
		int intermediateSize = 3072;
		int textEmbedDim = 512;
		
		ClipText clip = new ClipText(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, hiddenSize, textEmbedDim, maxPositionEmbeddingsSize, typeVocabSize, intermediateSize, numHiddenLayers);
		clip.CUDNN = true;
		clip.time = time;
		clip.RUN_MODEL = RunModel.EVAL;
		
		String clipWeight = "H:\\model\\clip_cn_vit-b-16.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
		
		int[] indexs = new int[] {0, 1};
		
		Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
		
		Tensor mask = new Tensor(batchSize , 1, 1, maxContextLen, true);
		
		dataLoader2.loadLabel(indexs, label, mask);
		
		Tensor input = new Tensor(batchSize, 3, imageSize, imageSize, true);
		
		dataLoader.loadData(indexs, input);
		
		JCudaDriver.cuCtxSynchronize();
		
//		Tensor out = network.forward(input);
		
		Tensor latent = network.encode(input);
		
//		latent.showShape();
		
//		latent.showDM();
		
		Tensor out = network.decode(latent);
		
		out.showShape();
//		out.showDM();
		
		out.syncHost();
		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
		
		/**
		 * print image
		 */
		MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test", mean, std);
		
		indexs = new int[] {4, 5, 6, 7};
		
		dataLoader.loadData(indexs, input);
		
		JCudaDriver.cuCtxSynchronize();
		
//		Tensor out = network.forward(input);
		Tensor clipOutput = null;
		for(int i = 0;i<10;i++) {
			long start = System.nanoTime();
			latent = network.encode(input);
			clipOutput = clip.forward(label, mask);
			out = network.decode(latent);
			JCuda.cudaDeviceSynchronize();
			System.err.println((System.nanoTime() - start)/1e6+"ms.");
		}
		
		
		out.showShape();
		clipOutput.showDM();
//		out.showDM();
		
		out.syncHost();
		out.data = MatrixOperation.clampSelf(out.data, -1, 1);
		
		/**
		 * print image
		 */
		MBSGDOptimizer.showImgs("H:\\vae_dataset\\pokemon-blip\\vqvae2\\test256\\", out, "test1", mean, std);
		
	}
	
	public static void test_clip() {
		
		String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
		
		String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
		String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
		
		boolean horizontalFilp = true;
		
		int imgSize = 256;
		
		int maxContextLen = 64;
		
		int batchSize = 4;

		float[] mean = new float[] {0.5f, 0.5f,0.5f};
		float[] std = new float[] {0.5f, 0.5f,0.5f};
		
		SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
		
		int time = maxContextLen;
		int maxPositionEmbeddingsSize = 512;
		int vocabSize = 21128;
		int hiddenSize = 768;
		int typeVocabSize = 2;
		int headNum = 12;
		int numHiddenLayers = 12;
		int intermediateSize = 3072;
		int textEmbedDim = 512;
		
		ClipText clip = new ClipText(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, hiddenSize, textEmbedDim, maxPositionEmbeddingsSize, typeVocabSize, intermediateSize, numHiddenLayers);
		clip.CUDNN = true;
		clip.time = time;
		clip.RUN_MODEL = RunModel.EVAL;
		
		String clipWeight = "H:\\model\\clip_cn_vit-b-16.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
		
		int[] indexs = new int[] {0, 1, 2, 3};
		
		Tensor label = new Tensor(batchSize * maxContextLen, 1, 1, 1, true);
		
		Tensor mask = new Tensor(batchSize , 1, 1, maxContextLen, true);
		
		dataLoader.loadLabel(indexs, label, mask);
		Tensor output = null;
		for(int i = 0;i < 100;i++) {
			long start = System.nanoTime();
			output = clip.forward(label, mask);
			JCuda.cudaDeviceSynchronize();
			System.err.println((System.nanoTime() - start)/1e6+"ms.");
			output.showShape();
//			output.showDM();
		}
		output.showDM();
		
	}
	
	public static void sd_train_pokem() throws Exception {
		
		String tokenizerPath = "H:\\clip\\CLIP\\clip_cn\\vocab.txt";
		
		String labelPath = "H:\\vae_dataset\\pokemon-blip\\data.json";
		String imgDirPath = "H:\\vae_dataset\\pokemon-blip\\dataset256\\";
		
		boolean horizontalFilp = true;
		
		int imgSize = 256;
		
		int maxContextLen = 64;
		
		int batchSize = 1;

		float[] mean = new float[] {0.5f, 0.5f,0.5f};
		float[] std = new float[] {0.5f, 0.5f,0.5f};
		
		SDImageDataLoader dataLoader = new SDImageDataLoader(tokenizerPath, labelPath, imgDirPath, imgSize, imgSize, maxContextLen, batchSize, horizontalFilp, mean, std);
		
		int imageSize = 256;
		int z_dims = 32;
		int latendDim = 4;
		
		int num_vq_embeddings = 512;
		
		int num_res_blocks = 2;
		
		int[] channels = new int[] {64, 128, 256};
		boolean[] attn_resolutions = new boolean[] {false, false, false};
		TinyVQVAE2 vae = new TinyVQVAE2(LossType.MSE, UpdaterType.adamw, z_dims, latendDim, num_vq_embeddings, imageSize, channels, attn_resolutions, num_res_blocks);
		vae.CUDNN = true;
		vae.learnRate = 0.001f;
		vae.RUN_MODEL = RunModel.EVAL;
		
		String vae_path = "H:\\model\\vqvae2_32_256_500.model";
		ModelUtils.loadModel(vae, vae_path);
		
		int time = maxContextLen;
		int maxPositionEmbeddingsSize = 512;
		int vocabSize = 21128;
		int hiddenSize = 768;
		int typeVocabSize = 2;
		int headNum = 12;
		int numHiddenLayers = 12;
		int intermediateSize = 3072;
		int textEmbedDim = 512;
		
		ClipText clip = new ClipText(LossType.MSE, UpdaterType.adamw, headNum, time, vocabSize, hiddenSize, textEmbedDim, maxPositionEmbeddingsSize, typeVocabSize, intermediateSize, numHiddenLayers);
		clip.CUDNN = true;
		clip.time = time;
		clip.RUN_MODEL = RunModel.EVAL;
		
		String clipWeight = "H:\\model\\clip_cn_vit-b-16.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), clip, true);
		
		int convOutChannels = 128;
		int unetHeadNum = 16;
		
		int[] downChannels = new int[] {32, 48, 64, 96};
		int[] midChannels = new int[] {96, 64};
		int numDowns = 1;
		int numMids = 1;
		int numUps = 1;
		
		boolean[] attns = new boolean[] {true, true, true};
		boolean[] downSamples = new boolean[] {true, true, true};
		
		int timeSteps = 1000;
		int tEmbDim = 512;
		
		DiffusionUNetCond unet = new DiffusionUNetCond(LossType.MSE, UpdaterType.adamw, latendDim, z_dims, z_dims, convOutChannels, unetHeadNum, downChannels, midChannels, downSamples, numDowns, numMids, numUps, timeSteps, tEmbDim, textEmbedDim, maxContextLen, attns);
		unet.CUDNN = true;
		unet.learnRate = 0.001f;
		
		MBSGDOptimizer optimizer = new MBSGDOptimizer(unet, 500, 0.00001f, batchSize, LearnRateUpdate.CONSTANT, false);
		
		optimizer.trainSD(dataLoader, vae, clip);
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			sd_train_pokem();
			
//			test_vqvae();
			
//			test_vqvae32();
			
//			test_clip();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
