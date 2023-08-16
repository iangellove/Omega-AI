package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.runtime.JCuda;

public class CUDAMemoryManager {
	
	public static Map<String,CUdeviceptr> deviceMap = new HashMap<String, CUdeviceptr>();
	
	public static Map<String,Pointer> pointerMap = new HashMap<String, Pointer>();
	
	public static List<CUdeviceptr> cu_deviceptrs = new ArrayList<CUdeviceptr>();
	
	public static List<Pointer> cu_porints = new ArrayList<Pointer>();
	
	public static GPUWorkspace workspace = new GPUWorkspace();
	
	public static CUdeviceptr getDevice(int size) {

		CUdeviceptr device = new CUdeviceptr();
		
		cuMemAlloc(device, size * Sizeof.FLOAT);
		
		cu_deviceptrs.add(device);
		
		return device;
	}
	
	public static CUdeviceptr getDevice(String key,int size) {
		
		if(deviceMap.containsKey(key)) {
			return deviceMap.get(key);
		}
		
		CUdeviceptr device = new CUdeviceptr();
//		System.out.println(key+":"+size);
		cuMemAlloc(device, size * Sizeof.FLOAT);
		
		deviceMap.put(key, device);
		
		return device;
	}
	
	public static Pointer getWorkspace(int size) {
		
		if(workspace.getSize() < size * Sizeof.FLOAT) {
			GPUOP.getInstance().free(workspace.getPointer());
			cudaMalloc(workspace.getPointer(), size * Sizeof.FLOAT);
			workspace.setSize(size * Sizeof.FLOAT);
		}
		
		return workspace.getPointer();
	}
	
	public static Pointer getPointer(int size) {
		Pointer p = new Pointer();
		cudaMalloc(p, size * Sizeof.FLOAT);
		cu_porints.add(p);
		return p;
	}
	
	public static Pointer getPointer(int size,int type) {

		Pointer p = new Pointer();
		cudaMalloc(p, size * type);

		cu_porints.add(p);
		
		return p;
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
		}
		
		for(String key:pointerMap.keySet()) {
			 GPUOP.getInstance().free(pointerMap.get(key));
		}

	}
	
	public static void freeAll() throws Exception{
		
		for(CUdeviceptr dec:cu_deviceptrs) {
			 JCuda.cudaFree(dec);
		}
		
		for(Pointer p:cu_porints) {
			 GPUOP.getInstance().free(p);
		}

	}
	
}
