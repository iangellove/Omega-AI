package com.omega.engine.gpu;

import com.omega.common.utils.JsonUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class BaseKernel {
	
	public int N = 0;
	
	public void showDM(Pointer p,float[] data) {
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void showDM(Pointer p,int[] data) {
		JCuda.cudaMemcpy(Pointer.to(data), p, data.length * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	    System.out.println(JsonUtils.toJson(data));
	}
	
}
