package com.omega.yolo.data;

import java.util.Map;

import com.omega.common.data.Tensor;

public abstract class DataTransform {
	
	public abstract void transform(Tensor input,Tensor label,String[] idxSet,int[] indexs,Map<String,float[]> orgLabelData);
	
	public abstract void showTransform(String outputPath,Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData);
	
}
