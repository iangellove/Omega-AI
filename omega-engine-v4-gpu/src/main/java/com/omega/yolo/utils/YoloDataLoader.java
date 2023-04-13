package com.omega.yolo.utils;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MathUtils;
import com.omega.engine.nn.data.DataSet;

/**
 * DataLoader
 * @author Administrator
 *
 */
public class YoloDataLoader extends BaseDataLoader{
	
	private String imgDirPath;
	
	private String[] idxSet;
	
	private Tensor imgSet;
	
	private Tensor labelSet;

	public YoloDataLoader(String imgDirPath,String labelPath,int batchSize,int channel ,int height,int width,int labelSize,LabelType mode,boolean normalization) {
		this.batchSize = batchSize;
		this.imgDirPath = imgDirPath;
		this.labelSize = labelSize;
		switch (mode) {
		case text:
			loadDataIdx(imgDirPath, labelPath);
			break;
		case csv:
			loadDataIdxFromCSV(imgDirPath, labelPath, channel, width, height, labelSize, normalization);
			break;
		default:
			break;
		}
		
	}
	
	public DataSet getDataSet() {
		return new DataSet(number, imgSet.channel, imgSet.height, imgSet.width, labelSet.channel, labelSize, imgSet.data, labelSet.data);
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
				
				setLabelSet(new Tensor(files.length, 1, 1, labelSize));
				
				LabelUtils.loadLabel(labelPath, getLabelSet());
				
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
	}
	
	public void loadDataIdxFromCSV(String imgDirPath,String label_csv_Path,int channel,int width,int height,int labelSize,boolean normalization) {
		try {
			
			File file = new File(imgDirPath);
			
			if(file.exists() && file.isDirectory()) {
				
				File[] files = file.listFiles();
				
				number = files.length;
				
				idxSet = new String[files.length];
				
				setLabelSet(new Tensor(files.length, 1, 1, labelSize));
				
				LabelUtils.loadLabelCSV(label_csv_Path, getLabelSet(), idxSet);
				
				setImgSet(new Tensor(files.length, channel, height, width));
				
				int index = 0;
				
				for(String filename:idxSet) {
					
					String filePath = imgDirPath + "/" + filename;
					
					YoloImageUtils.loadImgDataToTensor(filePath, getImgSet(), index, normalization);
					
					index++;
				}
				
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
			
			System.arraycopy(this.getLabelSet().data, index * labelSize, label.data, i * labelSize, labelSize);
		}
		
	}

	public Tensor getImgSet() {
		return imgSet;
	}

	public void setImgSet(Tensor imgSet) {
		this.imgSet = imgSet;
	}

	public Tensor getLabelSet() {
		return labelSet;
	}

	public void setLabelSet(Tensor labelSet) {
		this.labelSet = labelSet;
	}
	
}
