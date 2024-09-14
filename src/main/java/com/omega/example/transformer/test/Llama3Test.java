package com.omega.example.transformer.test;

import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.CNTokenizer;
import com.omega.example.transformer.utils.ModelUtils;
import com.omega.example.transformer.utils.bpe.BPETokenizer3;
import com.omega.example.transformer.utils.bpe.CNBpeTokenizer;

public class Llama3Test {
	
	
	public static void llama3_dp() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 32;
			
			int max_len = 128;
			
			int embedDim = 128;
			
			int headNum = 8;
			
			int nKVHeadNum = 4;
			
			int decoderNum = 8;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\dpcc50.txt";
			
			CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);

			Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, headNum, nKVHeadNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
			
//			network.CUDNN = true;
			
			network.learnRate = 0.001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 2, 0.001f, LearnRateUpdate.GD_GECAY, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainLlama3_GEN(trainData);
			
			int gen_len = 1000;
			
			network.RUN_MODEL = RunModel.TEST;
			
			Tensor input = null;
			
			Tensor output = null;
			
			String pre_txt = "萧炎";

			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			
			Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);

			for(int i = 0;i<gen_len;i++) {
				network.time = input.number;
//				System.out.println(input.number);
//				input.showDM();
				String txt = genTxt(input, output, network, trainData, pre_txt.length(), pos);
//				System.out.println("output txt="+txt);
				if(network.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
//				System.out.println(pre_txt);
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, max_len);
			}
			System.out.println(pre_txt);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void llama3_monkey() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			boolean flashAttention = false;
			
			int batchSize = 8;
			
			int max_len = 512;
			
			int embedDim = 512;
			
			int head_num = 16;
			
			int nKVHeadNum = 8;
			
			int decoderNum = 8;
			
			String trainPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.txt";
			
			String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
			
			String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
			
			BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
			
			CNBpeTokenizer trainData = new CNBpeTokenizer(trainPath, max_len, batchSize, 6250865, tokenizer);
			
			Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
			
			network.learnRate = 3e-4f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.COSINE, false);
			optimizer.lr_step = new int[] {1, 2};
			optimizer.lr = 3e-4f;
			optimizer.min_lr = 1e-5f;
			optimizer.setWarmUp(true);
			optimizer.warmUpTime = 1000;
			optimizer.lrDecayIters = (int) (trainData.count_it * 0.96);
			optimizer.trainLlama3_chinese(trainData);

			String model_path = "H:\\model\\llama3-26m-chinese.model";
	    
			ModelUtils.saveModel(network, model_path);
			
			network.RUN_MODEL = RunModel.TEST;
			Scanner scanner = new Scanner(System.in);
			
			while (true) {
				System.out.println("请输入中文:");
				String input_txt = scanner.nextLine();
				if(input_txt.equals("exit")){
					break;
				}
				input_txt = input_txt.toLowerCase();
				System.out.println("user:"+input_txt);
				int[] idx = tokenizer.encodeInt(input_txt);
				int startLen = idx.length;
				Tensor input = trainData.loadByTxtToIdx(idx);
//				input.showDM();
				Tensor[] pos = RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum);

				for(int t = 0;t<max_len - startLen;t++) {
					network.time = input.number;
					Tensor cos = pos[0];
					Tensor sin = pos[1];
					Tensor output = network.forward(cos, sin, input);
					output.syncHost();
					int nextIDX = output2NextIDX(output, idx.length - 1);
					idx = Arrays.copyOf(idx, idx.length + 1);
					idx[idx.length - 1] = nextIDX;
					if(nextIDX == tokenizer.eos) {
						break;
					}
					input = trainData.loadByTxtToIdx(idx);
					RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
				}
				System.out.println("chatbot:"+tokenizer.decode(idx));
			}
			scanner.close();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static String genTxt(Tensor input,Tensor output,Llama3 network,CNTokenizer trainData,int time,Tensor[] pos) {

		network.time = input.number;

		RoPEKernel.getCosAndSin(input.number, network.embedDim, network.headNum, pos);
		
		output = network.forward(pos[0], pos[1], input);
		output.syncHost();
		return output2TXT(output, trainData);
	}
	
	public static String output2TXT(Tensor output,CNTokenizer trainData) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 1);
			char c = trainData.dictionaryData[charIndex];
			txt += c;
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
	
	public static Tensor createTxtData(Tensor input,String txt,int charDim,Map<Character,Integer> dictionary,int maxLenght) {
		int charLength = txt.length();
		if(txt.length() > maxLenght) {
			charLength = maxLenght;
		}
		char[] charset = new char[charLength];
		int start = txt.length() - maxLenght;
		if(start <= 0) {
			start = 0;
		}
		txt.getChars(start, txt.length(), charset, 0);

		float[] td = new float[charLength];
		
		for(int i = 0;i<charLength;i++) {
			td[i] = dictionary.get(charset[i]);
		}
		if(input == null || input.number != charset.length){
			input = Tensor.createTensor(input, charset.length, 1, 1, 1, td, true);
		}else {
			input.data = td;
			input.hostToDevice();
		}
		return input;
	}
	
	public static int output2NextIDX(Tensor output,int nextTokenIdx) {
		if(nextTokenIdx < output.number) {
			return pickTopN(output.getByNumber(nextTokenIdx), 3);
		}
		return 0;
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
//			llama3_dp();
			
			llama3_monkey();

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
	}
	
}
