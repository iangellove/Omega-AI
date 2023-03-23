package com.omega.yolo.utils;

import com.omega.common.data.Tensor;

public abstract class BaseDataLoader {
	
	public int number;
	
	public int batchSize;
	
	public int labelSize;
	
	public abstract int[][] shuffle();
	
	public abstract void loadData(int[] indexs,Tensor input,Tensor label);
	
}
