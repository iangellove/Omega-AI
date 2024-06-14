package com.omega.example.transformer.test;

import java.util.Arrays;
import java.util.Scanner;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.NanoGPT;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.CNChatTokenizer;
import com.omega.example.transformer.utils.CNTokenizer;

public class Llama2Test {
	
	public static void llama2_yl_qa() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 16;
			
			int max_len = 128;
			
			int embedDim = 512;
			
			int head_num = 8;
			
			int decoderNum = 8;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\qaData.txt";

			CNChatTokenizer trainData = new CNChatTokenizer(trainPath, max_len, batchSize);
			
			Llama2 network = new Llama2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.SMART_HALF, false);
//			optimizer.lr_step = new int[] {1, 2};
			optimizer.trainLlama2(trainData);

//			network.RUN_MODEL = RunModel.TEST;
//			Scanner scanner = new Scanner(System.in);
//			while (true) {
//				System.out.println("请输入中文:");
//				String input_txt = scanner.nextLine();
//				if(input_txt.equals("exit")){
//					break;
//				}
//				input_txt = input_txt.toLowerCase() + " ";
//				System.out.println("user:"+input_txt);
//				Tensor input = trainData.loadByTxtToIdx(input_txt);
////				input.showDM();
//				Tensor positions = CNChatTokenizer.getPositions(1, input.number);
////				positions.showDM();
////				Tensor mask = CNChatTokenizer.triu(1, network.headNum, input.number, input.number, 1);
////				mask.showDM();
//				for(int t = 0;t<max_len;t++) {
//					network.time = input.number;
//					Tensor output = network.forward(input, positions);
//					output.syncHost();
////					output.showDM();
//					String txts = output2TXT(output, trainData, true);
////					System.out.println("output:"+txts);
//					String nextWord = txts.substring(txts.length() - 1, input_txt.length());
////					System.out.println("nextWord:"+nextWord);
//					
//					if(trainData.sd.get(nextWord)!=null && (trainData.sd.get(nextWord).equals("<sep>") || trainData.sd.get(nextWord).equals("<eos>"))) {
//						input_txt += trainData.sd.get(nextWord);
//						break;
//					}else {
//						input_txt += nextWord;
//					}
//					input = trainData.loadByTxtToIdx(input_txt);
//					CNChatTokenizer.getPositions(1, input.number, positions);
//					
////					CNChatTokenizer.triu(1, network.headNum, input.number, input.number, 1, mask);
//				}
//				
//				System.out.println("chatbot:"+input_txt.split(" ")[1]);
//			}
//			scanner.close();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void llama2_dp() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = true;
			
			int batchSize = 32;
			
			int max_len = 64;
			
			int embedDim = 128;
			
			int headNum = 8;
			
			int decoderNum = 8;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\dpcc50.txt";
			
			CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);

			Llama2 network = new Llama2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, headNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
			
//			network.CUDNN = true;
			
			network.learnRate = 0.001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 3, 0.001f, LearnRateUpdate.GD_GECAY, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainLlama2_GEN(trainData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static String output2TXT(Tensor output,CNChatTokenizer trainData,boolean format) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 1);
			String c = trainData.vocab[charIndex];
			txt += c;
		}
//		System.out.println("output txt:"+txt);
		if(format) {
			for(String key:trainData.specials_dictionary.keySet()) {
				txt = txt.replaceAll(key, trainData.specials_dictionary.get(key));
			}
		}
		return txt;
	}
	
	public static int pickTopN(float[] x,int n) {

		float[] sort = Arrays.copyOf(x, x.length);
		
		Arrays.sort(sort);
		
		float[] topN = Arrays.copyOfRange(sort, sort.length - n, sort.length);
		
		float v = topN[RandomUtils.getRandomNumber(topN)];
		
		for(int i = 0;i<x.length;i++) {
			if(v == x[i]) {
				return i;
			}
		}
		
		return 0;
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
//			llama2_yl_qa();
			
			llama2_dp();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
}
