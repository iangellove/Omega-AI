package com.omega.yolo.utils;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MathUtils;

/**
 * DataLoader
 * @author Administrator
 *
 */
public class YoloDataLoader extends BaseDataLoader{
	
	private String imgDirPath;
	
	private String[] idxSet;
	
	private Tensor labelSet;

	public YoloDataLoader(String imgDirPath,String labelPath,int batchSize,int channel ,int height,int width,int labelSize) {
		this.batchSize = batchSize;
		this.imgDirPath = imgDirPath;
		this.labelSize = labelSize;
		loadDataIdx(imgDirPath, labelPath);
	}
	
	public void loadDataIdx(String imgDirPath,String labelPath) {
		try {
			
			File file = new File(imgDirPath);
			
			if(file.exists() && file.isDirectory()) {
				
				File[] files = file.listFiles();
				
				number = files.length;
				
				idxSet = new String[files.length];
				
				for(int i = 0;i<files.length;i++) {
					
					File img = files[i];
					
					String filename = img.getName();
					
					idxSet[i] = filename;
				}
				
				orderByName();
				
				labelSet = new Tensor(files.length, 1, 1, labelSize);
				
				LabelUtils.loadLabel(labelPath, labelSet);
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	
	public void orderByName() {
	    List<String> list = Arrays.asList(idxSet);
	    Collections.sort(list, new Comparator<String>() {
	        @Override
	        public int compare(String o1, String o2) {
	            return o1.compareTo(o2);
	        }
	    });
	    idxSet = (String[]) list.toArray();
	}
	
	public int[][] shuffle(){
		return MathUtils.randomInts(this.number,this.batchSize);
	}
	
	public void loadData(int[] indexs,Tensor input,Tensor label) {

		for(int i = 0;i<indexs.length;i++) {
			int index = indexs[i];
			String filePath = imgDirPath + "/" + idxSet[index];
			
			YoloImageUtils.loadImgDataToTensor(filePath, input, i);
			
			System.arraycopy(this.labelSet.data, index * labelSize, label.data, i * labelSize, labelSize);
		}
		
	}
	
}
