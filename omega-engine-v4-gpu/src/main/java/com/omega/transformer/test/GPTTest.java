package com.omega.transformer.test;

import java.util.Arrays;
import java.util.Map;
import java.util.Scanner;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.EmbeddingLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.RNNBlockLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.GPT;
import com.omega.engine.nn.network.GPT2;
import com.omega.engine.nn.network.NanoGPT;
import com.omega.engine.nn.network.RNN;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.MBSGDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.rnn.data.OneHotDataLoader;
import com.omega.transformer.utils.CNChatTokenizer;
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

			CNChatTokenizer trainData = new CNChatTokenizer(trainPath, max_len, batchSize);
			
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
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 32;
			
			int max_len = 128;
			
			int embedDim = 512;
			
			int head_num = 6;
			
			int decoderNum = 4;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\chatdata\\train-format1w.txt";

			CNChatTokenizer trainData = new CNChatTokenizer(trainPath, max_len, batchSize);
			
			NanoGPT network = new NanoGPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, head_num, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, false);
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 1, 0.0001f, LearnRateUpdate.SMART_HALF, false);
			optimizer.lr_step = new int[] {200, 300, 500, 600, 700, 800, 900};
			optimizer.trainNanoGPT(trainData);

			Tensor positions = CNChatTokenizer.getPositions(1, max_len);
			
			Tensor mask = CNChatTokenizer.triu(1, network.headNum, network.time, network.time, 1);
			
			Scanner scanner = new Scanner(System.in);
			while (true) {
				System.out.println("请输入中文:");
				String input_txt = scanner.nextLine();
				if(input_txt.equals("exit")){
					break;
				}
				input_txt = input_txt.toLowerCase() + " ";
				System.out.println("input_txt:"+input_txt);
				Tensor input = trainData.loadByTxt(input_txt);
				for(int t = 0;t<max_len;t++) {
					Tensor output = network.forward(input, positions, mask);
					output.syncHost();
					String txts = output2TXT(output, trainData, true);
//					System.out.println("output:"+txts);
					String nextWord = txts.substring(input_txt.length(), input_txt.length() + 1);
					System.out.println("nextWord:"+nextWord);
					
					if(trainData.sd.get(nextWord)!=null && (trainData.sd.get(nextWord).equals("<sep>") || trainData.sd.get(nextWord).equals("<eos>"))) {
						input_txt += trainData.sd.get(nextWord);
						break;
					}else {
						input_txt += nextWord;
					}
					input = trainData.loadByTxt(input_txt);
				}
				
				System.out.println(input_txt);
			}
			scanner.close();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void gpt2_lang() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 10;
			
			int max_len = 256;
			
			int embedDim = 128;
			
			int headNum = 8;
			
			int decoderNUm = 6;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\lang\\lang.txt";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			GPT2 network = new GPT2(LossType.softmax_with_cross_entropy, UpdaterType.adamw, decoderNUm, headNum, trainData.vocab_size, max_len, embedDim, bias, dropout);
			
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
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 12;
			
			int max_len = 128;
			
			int embedDim = 512;
			
			int headNum = 6;
			
			int decoderNUm = 4;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\wikitext-2-v1\\wikitext-2\\wiki.train.tokens";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);
			
			NanoGPT network = new NanoGPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, decoderNUm, headNum, trainData.vocab_size, max_len, embedDim, bias, dropout);
			
			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.001f, LearnRateUpdate.CONSTANT, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainNanoGPT(trainData);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void nano_gpt_lang() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 10;
			
			int max_len = 128;
			
			int embedDim = 512;
			
			int headNum = 6;
			
			int decoderNum = 4;
			
			String trainPath = "H:\\transformer_dataset\\gpt\\lang\\lang.txt";

			ENTokenizer trainData = new ENTokenizer(trainPath, max_len, batchSize);

			NanoGPT network = new NanoGPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, headNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout);
			
//			network.CUDNN = true;
			
			network.learnRate = 0.0001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 300, 0.001f, LearnRateUpdate.CONSTANT, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainNanoGPT(trainData);

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
	
	public static void gpt_dp() {
		
		try {
			
			boolean bias = false;
			
			boolean dropout = false;
			
			int batchSize = 32;
			
			int max_len = 128;
			
			int embedDim = 128;
			
			int headNum = 4;
			
			int decoderNum = 4;
			
			String trainPath = "H:\\rnn_dataset\\dpcc.txt";
			
			CNTokenizer trainData = new CNTokenizer(trainPath, max_len, batchSize);

			NanoGPT network = new NanoGPT(LossType.softmax_with_cross_entropy, UpdaterType.adamw, headNum, decoderNum, trainData.characters, max_len, embedDim, bias, dropout);
			
//			network.CUDNN = true;
			
			network.learnRate = 0.001f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 3, 0.001f, LearnRateUpdate.GD_GECAY, false);
//			optimizer.lr_step = new int[] {20,50,80};
			optimizer.trainNanoGPT_GEN(trainData);
			
			int gen_len = 1000;
			
			network.RUN_MODEL = RunModel.TEST;
			
			Tensor input = null;
			
			Tensor output = null;
			
			String pre_txt = "干 枯 的 手 掌 仓 皇 的 抓 着 正 疑 惑 的 纳 兰 嫣 然 以 及 那 名 青 年 ， 然 后 逃 命 般 的 窜 出 了 大 厅 之 中";
			
			Tensor positions = CNChatTokenizer.getPositions(1, pre_txt.length());
			
			Tensor mask = CNChatTokenizer.triu(1, network.headNum, pre_txt.length(), pre_txt.length(), 1);
			
			input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, pre_txt.length());
			
			for(int i = 0;i<gen_len;i++) {
				network.time = input.number;
				String txt = genTxt(input, output, network, trainData, pre_txt.length(), mask, positions);
				System.out.println("output txt="+txt);
				if(network.time > 1) {
					pre_txt += txt.substring(input.number - 1, input.number);
				}else {
					pre_txt += txt;
				}
				System.out.println(pre_txt);
				input = createTxtData(input, pre_txt, trainData.characters, trainData.dictionary, pre_txt.length());
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static String genTxt(Tensor input,Tensor output,RNN network,CNChatTokenizer trainData,int maxLength) {
		output = network.forward(input);
		output.syncHost();
//		output.showDMByNumber(0);
		return output2TXT(output, trainData);
	}
	
	public static String output2TXT(Tensor output,CNChatTokenizer trainData) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 3);
			String c = trainData.vocab[charIndex];
			txt += c;
		}
		return txt;
	}
	
	public static String output2TXT(Tensor output,CNChatTokenizer trainData,boolean format) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 1);
			String c = trainData.vocab[charIndex];
			txt += c;
		}
		System.out.println("output txt:"+txt);
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
//		System.out.println(JsonUtils.toJson(charset));
		float[] td = new float[charLength * charDim];
		
		for(int i = 0;i<charLength;i++) {
			td[i * charDim + dictionary.get(charset[i])] = 1;
		}
		input = Tensor.createTensor(input, charset.length, 1, 1, charDim, td, true);
		return input;
	}
	
	public static String genTxt(Tensor input,Tensor output,NanoGPT network,CNTokenizer trainData,int maxLength,Tensor mask, Tensor positions) {

		CNChatTokenizer.getPositions(1, maxLength, positions);
		
		CNChatTokenizer.triu(1, network.headNum, maxLength, maxLength, 1, mask);
		
		network.time = maxLength;

		output = network.forward(input, positions, mask);
		output.syncHost();
//		output.showDMByNumber(0);
		return output2TXT(output, trainData);
	}
	
	public static String output2TXT(Tensor output,CNTokenizer trainData) {
		String txt = "";
		for(int i = 0;i<output.number;i++) {
			int charIndex = pickTopN(output.getByNumber(i), 3);
			char c = trainData.dictionaryData[charIndex];
			txt += c;
		}
		return txt;
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
//			gpt();

//			gpt_lang();
			
//			ch_chat();
			
//			gpt2_lang();
			
//			ch_chat_gpt2();
			
			gpt_dp();
			
//			gpt2_gan();
			
//			nano_gpt_lang();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
