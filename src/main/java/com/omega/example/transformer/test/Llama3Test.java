package com.omega.example.transformer.test;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.layer.gpu.RoPEKernel;
import com.omega.engine.nn.layer.llama.LlamaCausalSelfAttentionLayer;
import com.omega.engine.nn.layer.llama.LlamaMLPLayer;
import com.omega.engine.nn.layer.llama.LlamaTransformerBlock;
import com.omega.engine.nn.network.DiffusionUNet;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.example.transformer.utils.CNTokenizer;
import com.omega.example.transformer.utils.LagJsonReader;
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
			
			int batchSize = 4;
			
			int max_len = 512;
			
			int embedDim = 512;
			
			int head_num = 8;
			
			int nKVHeadNum = 4;
			
			int decoderNum = 8;
			
			String trainPath = "H:\\transformer_dataset\\6400\\monkey_idx_6400_vocab.bin";
			
			String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
			
			String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt";
			
			BPETokenizer3 tokenizer = new BPETokenizer3(vocabPath, mergesPath);
			
			CNBpeTokenizer trainData = new CNBpeTokenizer(trainPath, max_len, batchSize, tokenizer);
			
			Llama3 network = new Llama3(LossType.softmax_with_cross_entropy_idx, UpdaterType.adamw, head_num, nKVHeadNum, decoderNum, trainData.vocab_size, max_len, embedDim, bias, dropout, flashAttention);
			
			network.learnRate = 2e-4f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 5, 0.0001f, LearnRateUpdate.SMART_HALF, false);
			optimizer.lr_step = new int[] {1, 2, 4};
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
	
	public static void loadWeight(Map<String, Object> weightMap,Llama3 network) {
		for(String key:weightMap.keySet()) {
			System.out.println(key);
		}
		
		loadData(network.getDecoder().getSrc_emb().weight, weightMap.get("tok_embeddings.weight"), "tok_embeddings.weight");
		
		for(int i = 0;i<8;i++) {
			LlamaTransformerBlock block = network.getDecoder().getDecoderLayers().get(i);
			LlamaCausalSelfAttentionLayer attn = (LlamaCausalSelfAttentionLayer) block.getAttn();
			loadData(attn.getqLinerLayer().weight, weightMap.get("layers."+i+".attention.wq.weight"), "layers."+i+".attention.wq.weight");
			loadData(attn.getkLinerLayer().weight, weightMap.get("layers."+i+".attention.wk.weight"), "layers."+i+".attention.wk.weight");
			loadData(attn.getvLinerLayer().weight, weightMap.get("layers."+i+".attention.wv.weight"), "layers."+i+".attention.wv.weight");
			loadData(attn.getoLinerLayer().weight, weightMap.get("layers."+i+".attention.wo.weight"), "layers."+i+".attention.wo.weight");
			block.getNorm1().gamma = loadData(block.getNorm1().gamma, weightMap.get("layers."+i+".attention_norm.weight"), 1, "layers."+i+".attention_norm.weight");
			
			block.getNorm2().gamma = loadData(block.getNorm2().gamma, weightMap.get("layers."+i+".ffn_norm.weight"), 1, "layers."+i+".ffn_norm.weight");
			LlamaMLPLayer mlp = block.getMlp();
			loadData(mlp.getLinear1().weight, weightMap.get("layers."+i+".feed_forward.w1.weight"), "layers."+i+".feed_forward.w1.weight");
			loadData(mlp.getLinear2().weight, weightMap.get("layers."+i+".feed_forward.w2.weight"), "layers."+i+".feed_forward.w2.weight");
			loadData(mlp.getLinear3().weight, weightMap.get("layers."+i+".feed_forward.w3.weight"), "layers."+i+".feed_forward.w3.weight");
		}
		
		network.getDecoder().getNorm().gamma = loadData(network.getDecoder().getNorm().gamma, weightMap.get("norm.weight"), 1, "norm.weight");
		loadData(network.getFullyLayer().weight, weightMap.get("output.weight"), "output.weight");
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
	
	public static void loadData(Tensor x,Object meta,String key) {
		
		if(meta!=null) {
			int dim = getDim(x);
			if(dim == 1) {
				List<Double> dataA = (List<Double>) meta;
				for(int n = 0;n<dataA.size();n++) {
					x.data[n] = dataA.get(n).floatValue();
				}
			}else if(dim == 2) {
				
				List<List<Double>> dataA = (List<List<Double>>) meta;
				x.showShape();
				System.out.println(dataA.size()+":"+dataA.get(0).size());
				for(int n = 0;n<dataA.size();n++) {
					for(int w = 0;w<dataA.get(n).size();w++) {
						x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
					}
				}

			}else if(dim == 3) {
				float[][][] data = (float[][][]) meta;
				x.data = MatrixUtils.transform(data);
			}else{
				List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
				int N = dataA.size();
				int C = dataA.get(0).size();
				int H = dataA.get(0).get(0).size();
				int W = dataA.get(0).get(0).get(0).size();

				for(int n = 0;n<N;n++) {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								x.data[n * x.getOnceSize() + c * H * W + h * W + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
							}
						}
					}
				}

			}
			x.hostToDevice();
			System.out.println(key+"_finish.");
		}
	}
	
	public static Tensor loadData(Tensor x,Object meta,int dim,String key) {
		if(meta!=null) {
			if(dim == 1) {
				List<Double> dataA = (List<Double>) meta;
				x = new Tensor(1, 1, 1, dataA.size(), true);
				for(int n = 0;n<dataA.size();n++) {
					x.data[n] = dataA.get(n).floatValue();
				}
			}else if(dim == 2) {
				List<List<Double>> dataA = (List<List<Double>>) meta;
				x = new Tensor(dataA.size(), 1, 1, dataA.get(0).size(), true);
				for(int n = 0;n<dataA.size();n++) {
					for(int w = 0;w<dataA.get(n).size();w++) {
						x.data[n * dataA.get(n).size() + w] = dataA.get(n).get(w).floatValue();
					}
				}
//				float[][] data = (float[][]) meta;
//				x.data = MatrixUtils.transform(data);
			}else if(dim == 3) {
				float[][][] data = (float[][][]) meta;
				x.data = MatrixUtils.transform(data);
			}else{
				List<List<List<List<Double>>>> dataA = (List<List<List<List<Double>>>>) meta;
				int N = dataA.size();
				int C = dataA.get(0).size();
				int H = dataA.get(0).get(0).size();
				int W = dataA.get(0).get(0).get(0).size();
				x = new Tensor(N, C, H, W, true);
				for(int n = 0;n<dataA.size();n++) {
					for(int c = 0;c<dataA.get(n).size();c++) {
						for(int h = 0;h<dataA.get(n).get(c).size();h++) {
							for(int w = 0;w<dataA.get(n).get(c).get(h).size();w++) {
								x.data[n * x.getOnceSize() + c * x.height * x.width + h * x.width + w] = dataA.get(n).get(c).get(h).get(w).floatValue();
							}
						}
					}
				}

			}
			x.hostToDevice();
			System.out.println(key+"_finish.");
			return x;
		}
		return null;
	}
	
	public static int getDim(Tensor x) {
		int dim = 0;
		if(x.number > 1) {
			dim++;
		}
		if(x.channel > 1) {
			dim++;
		}
		if(x.height > 1) {
			dim++;
		}
		if(x.width > 1) {
			dim++;
		}
		return dim;
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
