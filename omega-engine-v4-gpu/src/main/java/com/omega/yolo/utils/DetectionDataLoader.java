package com.omega.yolo.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;
import com.omega.yolo.data.DataTransform;
import com.omega.yolo.data.DataType;
import com.omega.yolo.data.FileDataLoader;

/**
 * DetectionDataLoader
 * @author Administrator
 *
 */
public class DetectionDataLoader extends BaseDataLoader{
	
	private String imgDirPath;
	
	private String labelPath;
	
	private Map<String,List<float[]>> orgLabelData;
	
	private boolean dataEnhance = false;
	
	private LabelFileType labelFileType;
	
	private String[] idxSet;
	
	private int onceLabelSize = 5;
	
	private int img_w;
	
	private int img_h;
	
	private DataType dataType;
	
	private int classNum = 1;
	
	private int stride = 7;
	
	private String extName;
	
	private DataTransform dataTransform;
	
	public static float[] mean = new float[] {0.491f, 0.482f, 0.446f};
	public static float[] std = new float[] {0.247f, 0.243f, 0.261f};
	
	public DetectionDataLoader(String imgDirPath,String labelPath,LabelFileType labelFileType,int img_w,int img_h,int classNum,int batchSize,DataType dataType) {
		this.imgDirPath = imgDirPath;
		this.labelPath = labelPath;
		this.labelFileType = labelFileType;
		this.img_w = img_w;
		this.img_h = img_h;
		this.dataType = dataType;
		this.classNum = classNum;
		this.batchSize = batchSize;
		loadFileCount();
		loadLabelData();
	}
	
	public DetectionDataLoader(String imgDirPath,String labelPath,LabelFileType labelFileType,int img_w,int img_h,int classNum,int batchSize,DataType dataType,int onceLabelSize) {
		this.imgDirPath = imgDirPath;
		this.labelPath = labelPath;
		this.labelFileType = labelFileType;
		this.onceLabelSize = onceLabelSize;
		this.img_w = img_w;
		this.img_h = img_h;
		this.batchSize = batchSize;
		this.dataType = dataType;
		this.classNum = classNum;
		loadFileCount();
		loadLabelData();
	}
	
	public DetectionDataLoader(String imgDirPath,String labelPath,LabelFileType labelFileType,int img_w,int img_h,int classNum,int batchSize,DataType dataType,DataTransform dataTransform) {
		this.imgDirPath = imgDirPath;
		this.labelPath = labelPath;
		this.labelFileType = labelFileType;
		if(dataTransform != null) {
			this.dataEnhance = true;
		}
		this.dataTransform = dataTransform;
		this.img_w = img_w;
		this.img_h = img_h;
		this.dataType = dataType;
		this.classNum = classNum;
		this.batchSize = batchSize;
		loadFileCount();
		loadLabelData();
	}
	
	public DetectionDataLoader(String imgDirPath,String labelPath,LabelFileType labelFileType,int img_w,int img_h,int classNum,int batchSize,DataType dataType,int onceLabelSize,DataTransform dataTransform) {
		this.imgDirPath = imgDirPath;
		this.labelPath = labelPath;
		this.labelFileType = labelFileType;
		this.onceLabelSize = onceLabelSize;
		if(dataTransform != null) {
			this.dataEnhance = true;
		}
		this.dataTransform = dataTransform;
		this.img_w = img_w;
		this.img_h = img_h;
		this.dataType = dataType;
		this.classNum = classNum;
		this.batchSize = batchSize;
		loadFileCount();
		loadLabelData();
	}
	
	public void loadFileCount() {
		
		try {

			File file = new File(imgDirPath);
			
			if(file.exists() && file.isDirectory()) {
				String[] filenames = file.list();
				this.number = filenames.length;
				this.orgLabelData = new HashMap<String, List<float[]>>(number);
				this.idxSet = new String[number];
				this.extName = filenames[0].split("\\.")[1];
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void loadLabelData() {
		
		switch (labelFileType) {
		case txt:
			loadLabelDataForTXT();
			break;
		case csv:
			loadLabelDataForCSV();
			break;
		default:
			break;
		}
		
	}
	
	public void loadLabelDataForCSV() {
		
		try (FileInputStream fin = new FileInputStream(labelPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){

			String strTmp = "";
			int idx = 0;
	        while((strTmp = buffReader.readLine())!=null){
	        	if(idx > 0) {
		        	String[] list = strTmp.split(",");
		        	List<float[]> once = new ArrayList<float[]>();
		        	this.orgLabelData.put(list[0], once);
		        	this.idxSet[idx] = list[0];
		        	int page = (list.length - 1) / this.onceLabelSize;
		        	for(int i = 0;i<page;i++) {
		        		float[] bbox = new float[this.onceLabelSize];
		        		for(int j = 0;j<this.onceLabelSize;j++) {
		        			bbox[j] = Float.parseFloat(list[i * this.onceLabelSize + j + 1]);
		        		}
		        		once.add(bbox);
		        	}
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public void loadLabelDataForTXT() {
		
		try (FileInputStream fin = new FileInputStream(this.labelPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;
	        while((strTmp = buffReader.readLine())!=null){
	        	String[] list = strTmp.split(" ");
	        	List<float[]> once = new ArrayList<float[]>();
	        	int page = (list.length - 1) / this.onceLabelSize;
	        	for(int i = 0;i<page;i++) {
	        		float[] bbox = new float[this.onceLabelSize];
	        		for(int j = 0;j<this.onceLabelSize;j++) {
	        			bbox[j] = Float.parseFloat(list[i * this.onceLabelSize + j + 1]);
	        		}
	        		once.add(bbox);
	        	}
	        	this.orgLabelData.put(list[0], once);
	        	this.idxSet[idx] = list[0];
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	@Override
	public int[][] shuffle() {
		// TODO Auto-generated method stub
		return MathUtils.randomInts(this.number,this.batchSize);
	}
	
	@Override
	public void loadData(int pageIndex, int batchSize, Tensor input, Tensor label) {
		// TODO Auto-generated method stub
		
		int[] indexs = getIndexsByAsc(pageIndex, batchSize);
		
		/**
		 * 根据数据类型加载标签数据
		 */
		loadLabelToDataType(indexs, label, dataType);
		
		/**
		 * 加载input数据
		 */
		FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, true);
		
	}

	@Override
	public float[] loadData(int index) {
		// TODO Auto-generated method stub
		String filePath = imgDirPath + "/" + idxSet[index];
		return YoloImageUtils.loadImgDataToArray(filePath);
	}
	
	@Override
	public void loadData(int[] indexs, Tensor input, Tensor label) {
		// TODO Auto-generated method stub
//		long start = System.currentTimeMillis();

		/**
		 * 加载input数据
		 */
		FileDataLoader.load(imgDirPath, extName, idxSet, indexs, input.number, input, dataEnhance);
		
		/**
		 * 数据增强
		 */
		if(dataEnhance) {
			dataTransform.transform(input, input, mean, std);
		}
		
		/**
		 * 根据数据类型加载标签数据
		 */
		loadLabelToDataType(indexs, label, dataType);
//		System.out.println((System.currentTimeMillis() - start));
//		System.out.println(JsonUtils.toJson(input.data));
	}
	
	public void fileDataLoader(int[] indexs,Tensor input) {
		for (int i = 0; i < batchSize; i++) {
			String filePath = imgDirPath + "/" + idxSet[indexs[i]] + "." + extName;
			YoloImageUtils.loadImgDataToTensor(filePath, input, i);
//			System.out.println(filePath);
		}
	}
	
	public void loadLabelToDataType(int[] indexs, Tensor label,DataType dataType) {
		
		switch (dataType) {
		case yolov1:
			loadLabelToYolov1(indexs, label);
			break;
		case yolov3:
			loadLabelToYolov3(indexs, label, 90);
			break;
		}
		
	}
	
	public void loadLabelToYolov1(int[] indexs, Tensor label) {
		
		int once = (5+classNum);
		
		int oneSize = stride * stride * once;
		
		label.data = new float[label.number * oneSize];
		
		for(int b = 0;b<label.number;b++) {
			
			String key = idxSet[indexs[b]];
			
			List<float[]> list = orgLabelData.get(key);
			
			for(int n = 0;n<list.size();n++) {
			
				float[] box = list.get(n);
				
				int clazz = (int) box[0] + 1;
				
				float cx = box[1] / img_w;
				float cy = box[2] / img_h;
				float w = box[3] / img_w;
				float h = box[4] / img_h;
				
				int gridx = (int)(cx * stride);
				int gridy = (int)(cy * stride);
				
				float px = cx * stride - gridx;
				float py = cy * stride - gridy;

				/**
				 * c1
				 */
				label.data[b * oneSize + gridy * stride * once + gridx * once + 0] = 1.0f;
				
				/**
				 * class
				 */
				label.data[b * oneSize + gridy * stride * once + gridx * once + clazz] = 1.0f;
				
				/**
				 * x1,y1,w1,h1
				 */
				label.data[b * oneSize + gridy * stride * once + gridx * once + classNum + 1] = px;
				label.data[b * oneSize + gridy * stride * once + gridx * once + classNum + 2] = py;
				label.data[b * oneSize + gridy * stride * once + gridx * once + classNum + 3] = w;
				label.data[b * oneSize + gridy * stride * once + gridx * once + classNum + 4] = h;
			
			}
		}
		
	}
	
	public void loadLabelToYolov3(int[] indexs, Tensor label,int bbox_num) {
		
		label.data = new float[label.data.length];
		
		for(int i = 0;i<label.number;i++) {

			String key = idxSet[indexs[i]];
			
			List<float[]> list = orgLabelData.get(key);
			
			for(int c = 0;c<list.size();c++) {
				
				float[] box = list.get(c);
				
				float clazz = box[0];
				
				float x1 = box[1];
				float y1 = box[2];
				float x2 = box[3];
				float y2 = box[4];
				
				float cx = (x1 + x2) / (2 * img_w);
				float cy = (y1 + y2) / (2 * img_h);
				
				float w = (x2 - x1) / img_w;
				float h = (y2 - y1) / img_h;
				
				label.data[i * bbox_num * 5 + c * 5 + 0] = cx;
				label.data[i * bbox_num * 5 + c * 5 + 1] = cy;
				label.data[i * bbox_num * 5 + c * 5 + 2] = w;
				label.data[i * bbox_num * 5 + c * 5 + 3] = h;
				label.data[i * bbox_num * 5 + c * 5 + 4] = clazz;
				
			}
			
		}
		
	}
	
	public Tensor initLabelTensor() {
		
		switch (dataType) {
		case yolov1:
			return new Tensor(batchSize, 1, 1, (5 + classNum) * stride * stride);
		case yolov3:
			return new Tensor(batchSize, 1, 1, 5 * 90);
		}
		
		return null;
	}
	
	public int[] getIndexsByAsc(int pageIndex, int batchSize) {
		
		int start = pageIndex * batchSize;
		
		int end = pageIndex * batchSize + batchSize;
		
		if(end > number) {
			start = start - (end - number);
		}
		
		int[] indexs = new int[batchSize];
		
		for(int i = 0;i<batchSize;i++){
			indexs[i] = start + i;
		}
		return indexs;
	}
	
	public int getStride() {
		return stride;
	}

	public void setStride(int stride) {
		this.stride = stride;
	} 
	
}
