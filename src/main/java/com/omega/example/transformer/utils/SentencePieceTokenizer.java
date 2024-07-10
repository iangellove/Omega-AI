package com.omega.example.transformer.utils;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;

import ai.djl.sentencepiece.SpProcessor;
import ai.djl.sentencepiece.SpTokenizer;
import ai.djl.sentencepiece.SpVocabulary;

public class SentencePieceTokenizer extends BaseTokenizer{
	
	/** SentencePiece tokenizer. */
    private final SpProcessor tokenizer;
    /** Unknown token (<unk>), default id 0. */
    public final int unk;
    /** BOS (beginning of sequence) token (<s>), default id 1. */
    public final int bos;
    /** EOS (end of sequence) token (</s>), default id 2. */
    public final int eos;
    
    private final String[] _patterns = new String[]{"\\'", "\\\"", "\\.", "<br />", "\\,", "\\(", "\\)", "\\!", "\\?", "\\;", "\\:", "\\s+", "\\r", "\n"};

	private final String[] _replacements = new String[] {" '  ", "", " . ", " ", " , ", " ( ", " ) ", " ! ", " ? ", " ", " ", " ","",""};

    public SentencePieceTokenizer(String path) throws IOException {
        System.out.println("Loading SentencePiece model from " + path);
        SpTokenizer model = new SpTokenizer(Paths.get(path));
        SpVocabulary voc = SpVocabulary.from(model);
        tokenizer = model.getProcessor();
//        for(int i = 0;i<20000;i++) {
//        	System.out.println(i+":"+voc.getToken(i));
//        }
        
        unk = (int) voc.getIndex("<unk>");
        bos = (int) voc.getIndex("<s>");
        eos = (int) voc.getIndex("</s>");
        System.out.println("UNK ID: "+unk+" | BOS ID: "+bos+" | EOS ID: "+eos);
    }
	
    public int[] encode(String text) {
        return encode(text, false, false);
    }
    
    public int[] encode(String text, boolean bos, boolean eos) {
        int[] t = tokenizer.encode(text);

        int length = t.length;
        if (bos) ++length;
        if (eos) ++length;
        int[] tokens = length > t.length ? new int[length] : t;

        if (bos) {
            tokens[0] = this.bos;
            System.arraycopy(t, 0, tokens, 1, t.length);
        } else {
            System.arraycopy(t, 0, tokens, 0, t.length);
        }

        if (eos) {
            tokens[length - 1] = this.eos;
        }

        return tokens;
    }
    
    public String decode(int[] tokens) {
        return tokenizer.decode(tokens);
    }
    
    public String[] tokenize(String text) {
        return tokenizer.tokenize(text);
    }
    
    public void encodeDataset(String dataPath,String outputPath) {
    	
    	try {

        	List<Map<String, String>> list = LagJsonReader.readJsonFileSamll(dataPath);
    		
    		String strTmp = "";
    		
    		File file = new File(outputPath);
    		FileWriter writer = new FileWriter(file);
           
    		for(int i = 0;i<list.size();i++) {
    			strTmp = list.get(i).get("completion");	
    			for(int p = 0;p<_patterns.length;p++) {
            		strTmp = strTmp.replaceAll(_patterns[p], _replacements[p]);
            	}	
    			if(!strTmp.equals(" ") && !strTmp.equals("")) {
    				String idxStr = "";
    				int[] idx = encode(strTmp);
    				for(int id:idx) {
    					idxStr += id + " ";
    				}
    				writer.write(idxStr + "\n");
            	}
    			System.out.println(i);
    		}
        	
    		 writer.close();

             System.out.println("Data has been written to the file.");
             
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
    	
    }
    
    public static void main(String[] args) {
    	
//    	String tokenizer_path = "H:\\transformer_dataset\\tokenizer.model";
    	String tokenizer_path = "H:\\transformer_dataset\\chinese_sp.model";
    	
    	try {
			SentencePieceTokenizer t = new SentencePieceTokenizer(tokenizer_path);
			
			String txt = "中国社会科学院语言研究所是中国社会科学院下设的一个汉语语言研究机构。";
			
			String[] tokens = t.tokenize(txt);
			
			System.out.println(JsonUtils.toJson(tokens));
			
			int[] idx = t.encode(txt);
			
			System.out.println(JsonUtils.toJson(idx));
			
			String outText = t.decode(idx);
			
			System.out.println(outText);
			
//			String datasetPath = "H:\\transformer_dataset\\wikipedia-cn-20230720-filtered.json";
//			String outputPath = "H:\\transformer_dataset\\wiki_idx_2w_voc.txt";
//			
//			t.encodeDataset(datasetPath, outputPath);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

    }
    
}
