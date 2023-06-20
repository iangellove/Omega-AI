package com.omega.yolo.data;

import com.omega.common.data.Tensor;

public abstract class DataTransform {
	
	public abstract void transform(Tensor trainData,Tensor transData,float[] mean,float[] std);
	
}
