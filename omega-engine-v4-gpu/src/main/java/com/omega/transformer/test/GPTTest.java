package com.omega.transformer.test;

import java.util.Scanner;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.GPT;
import com.omega.engine.nn.network.Seq2SeqRNN;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.rnn.data.IndexDataLoader;
import com.omega.rnn.seq2seq.Seq2seq;
import com.omega.transformer.utils.ENTokenizer;

public class GPTTest {
	
	public static void gpt() {
		
		try {
			
			int batchSize = 12;
			
			int max_len = 256;
			
			int embedDim = 128;
			
			int nChannel = 512;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT network = new GPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, trainData.vocab_size, max_len, embedDim, nChannel);
			
			network.CUDNN = true;
			
			network.learnRate = 0.01f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 100, 0.001f, LearnRateUpdate.GD_GECAY, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainGPT(trainData);

//			Scanner scanner = new Scanner(System.in);
//			while (true) {
//				System.out.println("请输入英文:");
//				String input_txt = scanner.nextLine();
//				if(input_txt.equals("exit")){
//					break;
//				}
//				input_txt = input_txt.toLowerCase();
//				System.out.println(input_txt);
//				optimizer.predictRNN(trainData, input_txt);
//			}
//			scanner.close();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			gpt();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
