package com.omega.example.duffsion.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.data.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.example.yolo.utils.YoloImageUtils;

/**
 * DuffsionImageLoader
 * @author Administrator
 *
 */
public class DuffsionImageLoader extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6302699701667951010L;

	private int start = 0;
	
	private int end = 0;
	
	private int batchSize = 0;
	
	private String path;
	
	private String[] names;
	
	private int[] indexs;
	
	private Tensor input;
	
	private String extName;
	
	private static DuffsionImageLoader job;
	
	private boolean normalization = false;
	
	private float[] a;
	
	private float[] b;
	
	private float[] noise;
	
	private float[] mean = new float[] {0.5f, 0.5f, 0.5f};
	
	private float[] std = new float[] {0.5f, 0.5f, 0.5f};
	
	public static DuffsionImageLoader getInstance(String path,String extName,String[] names,int[] indexs,int batchSize,Tensor input,float[] a,float[] b,float[] noise,boolean normalization,int start,int end) {
		if(job == null) {
			job = new DuffsionImageLoader(path, extName, names, indexs, batchSize, input, a, b, noise, normalization, start, end);
		}else {
			if(input != job.getInput()){
				job.setInput(input);
			}
			job.setPath(path);
			job.setNames(names);
			job.setStart(0);
			job.setEnd(end);
			job.setIndexs(indexs);
			job.reinitialize();
		}
		return job;
	}
	
	public DuffsionImageLoader(String path,String extName,String[] names,int[] indexs,int batchSize,Tensor input,float[] a,float[] b,float[] noise,boolean normalization,int start,int end) {
		this.setStart(start);
		this.setEnd(end);
		this.batchSize = batchSize;
		this.setPath(path);
		this.extName = extName;
		this.setNames(names);
		this.setIndexs(indexs);
		this.setInput(input);
		this.a = a;
		this.b = b;
		this.noise = noise;
		this.normalization = normalization;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = getEnd() - getStart() + 1;
		
		if (length < 8 || length <= batchSize / 8) {
			
			load();

		} else {

			int mid = (getStart() + getEnd() + 1) >>> 1;
			DuffsionImageLoader left = new DuffsionImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), a, b, noise, normalization, getStart(), mid - 1);
			DuffsionImageLoader right = new DuffsionImageLoader(getPath(), extName, getNames(), getIndexs(), batchSize, getInput(), a, b, noise, normalization, mid, getEnd());

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}
	
	private void load() {
		
		for (int i = getStart(); i <= getEnd(); i++) {
			
			String filePath = getPath() + "/" + getNames()[getIndexs()[i]];
			
			if(!getNames()[getIndexs()[i]].contains(".")) {
				filePath = getPath() + "/" + getNames()[getIndexs()[i]] + "." + extName;
			}
			float[] data = YoloImageUtils.loadImgDataToArray(filePath, true, mean, std);
			for(int j = 0;j < data.length;j++) {
				data[j] = data[j] * a[i] + noise[i * data.length + j] * b[i];
			}
			System.arraycopy(data, 0, getInput().data, i * data.length, data.length);
		}
		
	}
	
	public static void load(String path,String extName,String[] names,int[] indexs,int batchSize,Tensor input,float[] a,float[] b,float[] noise,boolean normalization) {
		
//		FileDataLoader job = new FileDataLoader(path, extName, names, indexs, batchSize, input, 0, batchSize - 1);
		DuffsionImageLoader job = getInstance(path, extName, names, indexs, batchSize, input, a, b, noise, normalization, 0, batchSize - 1);
		ForkJobEngine.run(job);
		
	}

	public int getStart() {
		return start;
	}

	public void setStart(int start) {
		this.start = start;
	}

	public int getEnd() {
		return end;
	}

	public void setEnd(int end) {
		this.end = end;
	}

	public int[] getIndexs() {
		return indexs;
	}

	public void setIndexs(int[] indexs) {
		this.indexs = indexs;
	}

	public String getPath() {
		return path;
	}

	public void setPath(String path) {
		this.path = path;
	}

	public String[] getNames() {
		return names;
	}

	public void setNames(String[] names) {
		this.names = names;
	}

	public Tensor getInput() {
		return input;
	}

	public void setInput(Tensor input) {
		this.input = input;
	}
	
}
