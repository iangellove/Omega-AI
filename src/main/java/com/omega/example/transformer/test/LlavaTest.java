package com.omega.example.transformer.test;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.ClipVision;
import com.omega.engine.nn.network.Llava;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.clip.utils.ClipModelUtils;
import com.omega.example.transformer.dataset.LVMPreTrainDataset;
import com.omega.example.transformer.utils.LagJsonReader;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;

public class LlavaTest {
	
	public static ClipVision createClipVision() {
		
		boolean bias = true;
		
		int channel = 3;
		int imgSize = 224;
		int patchSize = 32;
		
		int headNum = 12;
		int nLayers = 12;
		int clip_time = 50;
		int embedDim = 768;
		
		ClipVision network = new ClipVision(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, channel, imgSize, patchSize, headNum, nLayers, clip_time, embedDim, bias, false);
		network.time = clip_time;
		network.CUDNN = true;
		
		String clipWeight = "H:\\model\\clip_vision_weights.json";
		ClipModelUtils.loadWeight(LagJsonReader.readJsonFileSmallWeight(clipWeight), network, false);
		System.out.println("clip vision is ready.");
		return network;
	}
	
	public static void llava_pertrain() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 2;
			
			int imageSize = 224;
			
			int max_len = 512;
			
			int imageTime = 50;
			
			int embedDim = 512;
			
			int visionOutDim = 768;
			
			int head_num = 16;
			
			int nKVHeadNum = 8;
			
			int decoderNum = 8;
			
			String trainPath = "I:\\BaiduNetdiskDownload\\dataset\\LLaVA-Pretrain\\chat-translated.json";
			
			String trainImagePath = "I:\\BaiduNetdiskDownload\\dataset\\pretrain_images\\";
			
			String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
			
			String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
			
			BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
			
			LVMPreTrainDataset trainData = new LVMPreTrainDataset(trainPath, trainImagePath, imageSize, max_len, batchSize, true, tokenizer);
			
			Llava network = new Llava(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, imageTime, embedDim, visionOutDim, bias, dropout);
			
			network.learnRate = 1e-4f;
			
			ClipVision clipVision = createClipVision();

			String model_path = "H:\\model\\llama3-26m-chinese.model";
			ModelUtils.loadPertrainModel(network, model_path);
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.0001f, LearnRateUpdate.CONSTANT, false);

			optimizer.train_llava_chinese(trainData, clipVision, 8, true);

			String save_model_path = "H:\\model\\llava-26m-chinese.model";
			ModelUtils.saveModel(network, save_model_path);

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args){
		
		try {

			CUDAModules.initContext();
			
			llava_pertrain();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
