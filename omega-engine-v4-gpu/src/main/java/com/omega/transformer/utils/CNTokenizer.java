package com.omega.transformer.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;

public class CNTokenizer extends BaseTokenizer{
	
	public int number = 0;
	
	private int batchSize = 1;
	
	private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+"};

	private final String[] _replacements = new String[] {" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " "};
	
	private String dataPath;
	
	public Map<String,Integer> dictionary = new HashMap<String, Integer>();
	
	private List<String> org_tokens = new ArrayList<String>();
	
	public List<String[]> tokens = new ArrayList<String[]>();
	
	private final String[] specials = new String[] {"<pad>","<sos>","<eos>","<sep>"};
	
	public int max_len = 256;
	
	public int vocab_size;
	
	public String[] vocab;
	
	public CNTokenizer(String dataPath,int max_len,int batchSize) {
		this.dataPath = dataPath;
		this.max_len = max_len;
		this.batchSize = batchSize;
		loadDataForTXT();
		this.number = org_tokens.size();
		System.out.println(this.number);
	}
	
	public void loadDataForTXT() {
		
		try (FileInputStream fin = new FileInputStream(this.dataPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
//			int dic_index = 0;
			String strTmp = "";
	        while((strTmp = buffReader.readLine())!=null){
	        	for(int i = 0;i<_patterns.length;i++) {
	        		strTmp = strTmp.replaceAll(_patterns[i], _replacements[i]);
	        	}
	        	strTmp = strTmp.toLowerCase();
	        	strTmp = strTmp.substring(0, strTmp.length() - 1);
	        	if(!strTmp.equals(" ") && !strTmp.equals("") && strTmp.length() <= max_len - 2) {
	        		org_tokens.add(strTmp);
//	        		strTmp = "<sos>" + strTmp + "<eos>";
//		        	System.out.println(strTmp);
//		        	System.out.println("line["+dic_index+"]:" + strTmp);
//		        	dic_index++;
	        	}
	        }
	        buildVocab();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void buildVocab() {
		for(int i = 0;i<specials.length;i++) {
			dictionary.put(specials[i], i);
		}
		int idx = specials.length;
		for(int i = 0;i<org_tokens.size();i++) {
			String[] once = org_tokens.get(i).split("");
//			System.out.println(JsonUtils.toJson(once));
			if(once.length > 1) {
				tokens.add(once);
				for(int j = 0;j<once.length;j++) {
					String txt = once[j];
					if(!txt.equals("")) {
						if(txt.equals(" ")) {
							txt = "<sep>";
							once[j] = "<sep>";
						}
						if(!dictionary.containsKey(txt)) {
		        			dictionary.put(txt, idx);
		        			idx++;
		        		}
					}
				}
			}
			
		}
		vocab_size = dictionary.size();
		vocab = new String[vocab_size];
		for(String key:dictionary.keySet()) {
			vocab[dictionary.get(key)] = key;
		}
	}
	
	public void loadData(int[] indexs, Tensor input, Tensor label) {
		// TODO Auto-generated method stub
		
		input.clear();
		label.clear();
		
		for(int i = 0;i<indexs.length;i++) {
			String[] onceToken = tokens.get(indexs[i]);
//			System.out.println(onceToken.length);
			for(int t = 0;t<max_len;t++) {
				format(i, t, onceToken, input, label);
			}
		}

		/**
		 * copy data to gpu.
		 */
		input.hostToDevice();
		label.hostToDevice();
		
	}
	
	public void format(int b,int t,String[] onceToken,Tensor input,Tensor label) {
		if(t == 0){
			String curr = onceToken[t];
			String next = onceToken[t+1];
			input.data[(b * max_len + t) * vocab_size + 1] = 1.0f;
			input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
			label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
			return;
		}
		if(t == onceToken.length - 1) {
			String curr = onceToken[t];
			input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
			label.data[(b * max_len + t) * vocab_size + 2] = 1.0f;
			return;
		}
		if((t + 1) < onceToken.length) {
			String curr = onceToken[t];
			String next = onceToken[t + 1];
//			System.out.println(next);
			input.data[(b * max_len + t + 1) * vocab_size + dictionary.get(curr)] = 1.0f;
			label.data[(b * max_len + t) * vocab_size + dictionary.get(next)] = 1.0f;
		}else {
			input.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
			label.data[(b * max_len + t) * vocab_size + 0] = 1.0f;
		}
	}

	
	public static Tensor getPositions(int b,int time) {
		float[] data = new float[b * time * time];
		for(int n = 0;n<b;n++) {
			for(int t = 0;t<time;t++) {
				data[n * time * time + t * time + t] = 1;
			}
		}
		Tensor positions = new Tensor(b, time, 1, time, data, true);
		
		return positions;
	}
	
	public static Tensor triu(int b,int h,int size1,int size2,float val) {
		float[] data = new float[b * h * size1 * size2];
		for(int n = 0;n<b;n++) {
			for(int hn = 0;hn<h;hn++) {
				for(int i = 0;i<size1;i++) {
					for(int j = 0;j<size2;j++) {
						if(i < j) {
							data[n * h * size1 * size2 + hn * size1 * size2 + i * size1 + j] = val;
						}
					}
				}
			}
		}
		
		Tensor mask = new Tensor(b, h, size1, size2, data, true);
		
		return mask;
	}
	
	public static void main(String[] args) {
		
//		String dataPath = "H:\\transformer_dataset\\gpt\\chatdata\\train1w.txt";
//		
//		int batchSize = 64;
//		
//		CNTokenizer tokenizer = new CNTokenizer(dataPath, 256, batchSize);
		
//		tokenizer.loadDataForTXT();
		
		Tensor subsequent_mask = triu(2, 4, 5, 5, 1);
		subsequent_mask.showDM();
		
//		Tensor positions = getPositions(2, 4);
//		positions.showDM();
	}
	
}