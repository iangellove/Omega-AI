package com.omega.yolo.utils;

import com.omega.common.data.Tensor;

public abstract class BaseDataLoader {
	
	public int number;
	
	public int batchSize;
	
	public int labelSize;
	
	public int labelChannel;
	
	public abstract int[][] shuffle();
	
	public abstract void loadData(int[] indexs,Tensor input,Tensor label);
	
	public abstract void loadData(int pageIndex,int batchSize,Tensor input,Tensor label);
	
}
