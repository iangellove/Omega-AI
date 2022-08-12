package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.HashMap;
import java.util.Map;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;

public class CUDAMemoryManager {
	
	public static Map<String,CUdeviceptr> deviceMap = new HashMap<String, CUdeviceptr>();
	
	public static Map<String,Pointer> pointerMap = new HashMap<String, Pointer>();
	
	public static CUdeviceptr getDevice(String key,int size) {
		
		if(deviceMap.containsKey(key)) {
			return deviceMap.get(key);
		}
		
		CUdeviceptr device = new CUdeviceptr();
		
		cuMemAlloc(device, size * Sizeof.FLOAT);
		
		deviceMap.put(key, device);
		
		return device;
	}
	
	public static Pointer getPointer(String key,int size) {
		
		if(pointerMap.containsKey(key)) {
			return pointerMap.get(key);
		}
		
		Pointer p = new Pointer();
		
		cudaMalloc(p, size * Sizeof.FLOAT);
		
		pointerMap.put(key, p);
		
		return p;
	}
	
	public static void free() {
		
		for(String key:deviceMap.keySet()) {
			 JCuda.cudaFree(deviceMap.get(key));
//			 deviceMap.remove(key);
		}
		
		for(String key:pointerMap.keySet()) {
			 GPUOP.getInstance().free(pointerMap.get(key));
//			 pointerMap.remove(key);
		}

	}
	
}
