package com.omega.example.transformer.utils.bpe;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.network.utils.ModelUtils;


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
		    	
		    	if(txt.length() <= 512) {

			    	if(txt != null && !txt.equals("")) {
			    		txtList.add(txt);
			    	}
			    	
			    	if(i > 1 && i % batchSize == 0) {
			    		EncodeEx.encode(txtList, ids, bpe);
			    		write(txtList, ids, writer, bpe);
			    		txtList.clear();
			    	}

	    			System.out.println(i);
	    			i++;
		    	}
		    	
		    }
			if(txtList.size() > 0) {
				EncodeEx.encode(txtList, ids, bpe);
	    		write(txtList, ids, writer, bpe);
			}
		    bufferedReader.close();
		    writer.close();
		} catch (IOException e) {
		    e.printStackTrace();
		}

        System.out.println("Data has been written to the file.");
         
    }
	
	public static void writeIn(List<String> txtList,String[] ids,FileWriter writer) throws IOException {
		System.out.println("writing.");
		for(int i = 0;i<txtList.size();i++) {
			String txt = ids[i];
			writer.write(txt + "\n");
		}
	}
	
	public static void write(List<String> txtList,String[] ids,FileWriter writer, BPETokenizer3 bpe) throws IOException {
		System.out.println("writing.");
		for(int i = 0;i<txtList.size();i++) {
			String txt = ids[i];
			writer.write(bpe.sos + " " + txt + " " + bpe.eos);
		}
	}
	
	public static void txt2bin(String txtPath,String binPath,int sos,int eos) {
		
		try {
			String line = null;
			FileInputStream fis = new FileInputStream(txtPath);
			BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
			
			File file = new File(binPath);
			if(!file.exists()) {
				try {
					file.createNewFile();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
			try(RandomAccessFile rFile = new RandomAccessFile(file, "rw")){
				
				int index = 0;
				while ((line = bufferedReader.readLine()) != null) {
					
					String[] txts = line.split(" ");
					
					int[] idx = new int[txts.length + 2];
					idx[0] = sos;
					idx[idx.length - 1] = eos;
					
					for(int i = 1;i<idx.length - 1;i++) {
						idx[i] = Integer.parseInt(txts[i - 1]);
					}
					
					ModelUtils.saveIntData(rFile, idx);
					index++;
					System.out.println(index);
				}
				
			}catch (Exception e) {
				// TODO: handle exception
				e.printStackTrace();
			}
			
			bufferedReader.close();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
//		String dataPath = "H:\\transformer_dataset\\mobvoi_seq_monkey_general_open_corpus\\mobvoi_seq_monkey_general_open_corpus.jsonl";
//		String outputPath = "H:\\transformer_dataset\\monkey_idx_6400_all_vocab.txt";
//		
		String vocabPath = "H:\\transformer_dataset\\6400\\vocab.json";
		String mergesPath = "H:\\transformer_dataset\\6400\\merges.txt"; 
//		
//		encodeMonkeyDataset(dataPath, outputPath, vocabPath, mergesPath);
		
//		String txtPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.txt";
		String binPath = "H:\\transformer_dataset\\monkey_idx_6400_vocab.bin";
		
//		txt2bin(txtPath, binPath, 1, 2);
		
		int time = 512;
		
		int[] data = new int[time];
		
		data = loadData(data, binPath);
		
		BPETokenizer3 bpe = new BPETokenizer3(vocabPath, mergesPath);
		
		String txt = bpe.decode(data);
		
		System.out.println(txt);
		
	}
	
	public static int[] loadData(int[] data,String inputPath) {
		
		try(RandomAccessFile file = new RandomAccessFile(inputPath, "r")){
			System.out.println(file.length() / 4 / 512);
			System.out.println(file.getFilePointer());
			ModelUtils.loadIntData(file, data);
			System.out.println(file.getFilePointer());
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return data;
	}
	
}
