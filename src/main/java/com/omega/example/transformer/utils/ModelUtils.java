package com.omega.example.transformer.utils;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.nn.network.Llama2;
import com.omega.engine.nn.network.Llama3;
import com.omega.engine.nn.network.Llava;
import com.omega.engine.nn.network.NanoGPT;
import com.omega.engine.nn.network.vae.TinyVQVAE;

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
	
	public static void saveModel(TinyVQVAE model,String outpath) {
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
	
	public static void loadModel(TinyVQVAE model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadModel(File);
			System.out.println("model load success...");
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
	
	public static void saveModel(Llama3 model,String outpath) {
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
	
	public static void loadModel(Llama3 model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadModel(File);
			System.out.println("model load success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void saveModel(Llava model,String outpath) {
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
	
	public static void loadModel(Llava model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadModel(File);
			System.out.println("model load success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}

	public static void loadPertrainModel(Llava model,String inputPath) {
		
		try(RandomAccessFile File = new RandomAccessFile(inputPath, "r")){
			System.out.println("start load model...");
			model.loadPertrainModel(File);
			initParams(model.getDecoder().getVersionProj().weight, 0.0f, 0.02f);
			System.out.println("model load success...");
		}catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void initParams(Tensor p,float mean,float std) {
		p.setData(RandomUtils.normal_(p.dataLength, mean, std));
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
