package com.omega.yolo.data;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.data.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.yolo.utils.YoloImageUtils;

/**
 * FileDataLoader
 * @author Administrator
 *
 */
public class DetectionDataLoader extends RecursiveAction {

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
	
	private DetectionLabelCreater labelCreater;
	
	private String extName;
	
	private static DetectionDataLoader job;
	
	private boolean normalization = false;

	public static DetectionDataLoader getInstance(String path,String extName,String[] names,DetectionLabelCreater labelCreater,int[] indexs,int batchSize,Tensor input,boolean normalization,int start,int end) {
		if(job == null) {
			job = new DetectionDataLoader(path, extName, names, labelCreater, indexs, batchSize, input, normalization, start, end);
		}else {
			job.setPath(path);
			job.setNames(names);
			job.setStart(0);
			job.setEnd(end);
			job.setIndexs(indexs);
			job.reinitialize();
			job.setLabelCreater(labelCreater);
		}
		return job;
	}
	
	public DetectionDataLoader(String path,String extName,String[] names,DetectionLabelCreater labelCreater,int[] indexs,int batchSize,Tensor input,boolean normalization,int start,int end) {
		this.setStart(start);
		this.setEnd(end);
		this.batchSize = batchSize;
		this.setPath(path);
		this.extName = extName;
		this.setNames(names);
		this.labelCreater = labelCreater;
		this.setIndexs(indexs);
		this.input = input;
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
			DetectionDataLoader left = new DetectionDataLoader(getPath(), extName, getNames(), labelCreater, getIndexs(), batchSize, input, normalization, getStart(), mid - 1);
			DetectionDataLoader right = new DetectionDataLoader(getPath(), extName, getNames(), labelCreater, getIndexs(), batchSize, input, normalization, mid, getEnd());

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}
	
	private void load() {
		
		for (int i = getStart(); i <= getEnd(); i++) {
			String key = names[indexs[i]];
			String filePath = getPath() + "/" + key + "." + extName;
			YoloImageUtils.loadImgDataToOrgTensor(filePath, input, i);
			loadLabel(key, i);
		}
		
	}
	
	private void loadLabel(String key,int index) {
		this.labelCreater.loadLabel(key, index);
	}
	
	public static void load(String path,String extName,String[] names,DetectionLabelCreater labelCreater,int[] indexs,int batchSize,Tensor input,boolean normalization) {
		labelCreater.clearData();
		DetectionDataLoader job = getInstance(path, extName, names, labelCreater, indexs, batchSize, input, normalization, 0, batchSize - 1);
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

	public void setLabelCreater(DetectionLabelCreater labelCreater) {
		this.labelCreater = labelCreater;
	}
	
}
