package com.omega.example.transformer.utils;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.NanoGPT;

public class ModelUtils {
	
	public static void saveModel(Llama2 model,String outpath) {
		File file = new File(outpath);
		if(!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		try(RandomAccessFile rFile = new RandomAccessFile(file, "rw")){
			System.out.println("start save model...");
			model.saveModel(rFile);
			System.out.println("model save success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void loadModel(Llama2 model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadModel(File);
			System.out.println("model load success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void saveModel(NanoGPT model,String outpath) {
		
		File file = new File(outpath);
		if(!file.exists()) {
			try {
				file.createNewFile();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		try(RandomAccessFile aFile = new RandomAccessFile(file, "rw")){
			System.out.println("start save model...");
			model.saveModel(aFile);
			System.out.println("model save success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void loadModel(NanoGPT model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadModel(File);
			System.out.println("model load success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
}
