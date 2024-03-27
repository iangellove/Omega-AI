package com.omega.transformer.test;

import java.util.Scanner;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.GPT;
import com.omega.engine.nn.network.GPT2;
import com.omega.engine.nn.network.Seq2SeqRNN;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.rnn.data.IndexDataLoader;
import com.omega.rnn.seq2seq.Seq2seq;
import com.omega.transformer.utils.CNTokenizer;
import com.omega.transformer.utils.ENTokenizer;

public class GPTTest {
	
	public static void gpt() {
		
		try {
			
			int batchSize = 16;
			
			int max_len = 256;
			
			int embedDim = 256;
			
			int nChannel = 1024;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT network = new GPT(LossType.softmax_with_cross_entropy, UpdaterType.adam, trainData.vocab_size, max_len, embedDim, nChannel);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
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
	
	public static void gpt_lang() {
		
		try {
			
			int batchSize = 10;
			
			int max_len = 256;
			
			int embedDim = 512;
			
			int nChannel = 2048;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\lang\\lang.txt";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT network = new GPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, trainData.vocab_size, max_len, embedDim, nChannel);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.001f, LearnRateUpdate.CONSTANT, false);
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
	
	public static void ch_chat() {
		
		try {
			
			int batchSize = 32;
			
			int max_len = 128;
			
			int embedDim = 768;
			
			int nChannel = 2048;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\chatdata\\train-format1w.txt";

			CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);
			
			GPT network = new GPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, trainData.vocab_size, max_len, embedDim, nChannel);
			
			network.CUDNN = true;
			
			network.learnRate = 0.01f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.0001f, LearnRateUpdate.GD_GECAY, false);
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
	
	public static void ch_chat_gpt2() {
		
		try {
			
			int batchSize = 32;
			
			int max_len = 128;
			
			int embedDim = 512;
			
			int head_num = 2;
			
			int decoderNum = 4;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\chatdata\\train-format1w.txt";

			CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);
			
			GPT2 network = new GPT2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, decoderNum, head_num, trainData.vocab_size, max_len, embedDim);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1000, 0.0001f, LearnRateUpdate.SMART_HALF, false);
			optimizer.lr_step = new int[] {300, 600};
			optimizer.trainGPT2(trainData);

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
	
	public static void gpt2_lang() {
		
		try {
			
			int batchSize = 10;
			
			int max_len = 256;
			
			int embedDim = 128;
			
			int headNum = 8;
			
			int decoderNUm = 6;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\lang\\lang.txt";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT2 network = new GPT2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, decoderNUm, headNum, trainData.vocab_size, max_len, embedDim);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.001f, LearnRateUpdate.CONSTANT, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainGPT2(trainData);

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
	
	public static void gpt2_gan() {
		
		try {
			
			int batchSize = 12;
			
			int max_len = 256;
			
			int embedDim = 512;
			
			int headNum = 4;
			
			int decoderNUm = 4;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT2 network = new GPT2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, decoderNUm, headNum, trainData.vocab_size, max_len, embedDim);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.001f, LearnRateUpdate.CONSTANT, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainGPT2(trainData);

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
			
//			gpt();

//			gpt_lang();
			
//			ch_chat();
			
//			gpt2_lang();
			
			ch_chat_gpt2();
			
//			gpt2_gan();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
