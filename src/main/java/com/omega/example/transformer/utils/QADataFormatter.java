package com.omega.example.transformer.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.utils.JsonUtils;

public class QADataFormatter {
	
	public static void loadQDataCSV(String csvPath,Map<String,String> qData) {
		
		try (FileInputStream fin = new FileInputStream(csvPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;

	        while((strTmp = buffReader.readLine())!=null){
//	        	System.out.println(strTmp);
	        	if(idx > 0) {
		        	String[] list = strTmp.split(",");
		        	qData.put(list[0], list[1]);
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void loadADataCSV(String csvPath,List<Map<String,String>> aData) {
		
		try (FileInputStream fin = new FileInputStream(csvPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;

	        while((strTmp = buffReader.readLine())!=null){
//	        	System.out.println(strTmp);
	        	if(idx > 0) {
		        	String[] list = strTmp.split(",");
		        	Map<String,String> once = new HashMap<String, String>();
		        	once.put("qid", list[1]);
		        	once.put("txt", list[2]);
		        	aData.add(once);
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void formatQAFormCSV(String questionPath,String answerPath,String outPath) {
		
		Map<String,String> question = new HashMap<String, String>();
		List<Map<String,String>> answer = new ArrayList<Map<String,String>>();
		
		loadQDataCSV(questionPath, question);
		loadADataCSV(answerPath, answer);
		
//		System.out.println(JsonUtils.toJson(answer));
		
		try (FileOutputStream fos = new FileOutputStream(outPath);) {

			for(Map<String,String> once:answer) {
				String qid = once.get("qid");
				String q = question.get(qid).replaceAll(" ", "");
				String a = once.get("txt").replaceAll(" ", "");
				String text = q + " " + a;
				text += "\n";
//					System.out.println(text);
				fos.write(text.getBytes());
			}
			fos.flush();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	
	public static void main(String[] args) {
		String questionPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\question\\question.csv";
		String answerPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\answer\\answer.csv";
		String outPath = "H:\\transformer_dataset\\gpt\\cMedQA2\\qaData.txt";
		
		formatQAFormCSV(questionPath, answerPath, outPath);
	}
	
}
