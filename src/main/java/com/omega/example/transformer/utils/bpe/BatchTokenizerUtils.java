package com.omega.example.transformer.utils.bpe;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;

public class BatchTokenizerUtils {
	
	
	public static void encodeMonkeyDataset(String dataPath,String outputPath,String vocabPath,String mergesPath) {

		try {
			File file = new File(outputPath);
    		FileWriter writer = new FileWriter(file);
			
			Map<String,String> once = new HashMap<String,String>();
			String line = null;
			FileInputStream fis = new FileInputStream(dataPath);
			BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
			
			BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);

			int batchSize = 10000;
			
			List<String> txtList = new ArrayList<String>();
			String[] ids = new String[batchSize];
			
			int i = 1;
			while ((line = bufferedReader.readLine()) != null) {
		    	once = JsonUtils.gson.fromJson(line, HashMap.class);
		    	
		    	String txt = once.get("text");
		    	
		    	if(txt != null && !txt.equals("")) {
		    		txtList.add(txt);
		    	}
		    	
		    	if(i > 1 && i % batchSize == 0) {
		    		EncodeEx.encode(txtList, ids, bpe);
		    		write(txtList, ids, writer);
		    		txtList.clear();
		    	}
		    	
    			System.out.println(i);
    			i++;
		    }
			if(txtList.size() > 0) {
				EncodeEx.encode(txtList, ids, bpe);
	    		write(txtList, ids, writer);
			}
		    bufferedReader.close();
		    writer.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}

        System.out.println("Data has been written to the file.");
         
    }
	
	public static void write(List<String> txtList,String[] ids,FileWriter writer) throws IOException {
		System.out.println("writing.");
		for(int i = 0;i<txtList.size();i++) {
			String txt = ids[i];
			writer.write(txt + "\n");
		}
	}
	
	public static void main(String[] args) {
		
		String dataPath = "H:\\transformer_dataset\\mobvoi_seq_monkey_general_open_corpus\\mobvoi_seq_monkey_general_open_corpus.jsonl";
		String outputPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.txt";
		
		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt"; 
		
		encodeMonkeyDataset(dataPath, outputPath, vocabPath, mergesPath);
		
	}
	
}
