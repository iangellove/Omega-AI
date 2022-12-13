package com.omega.common.data;

import java.io.Serializable;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.gpu.CUDAMemoryManager;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;

public class Tensor implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 5844762745177624845L;

	public int number = 0;
	
	public int channel = 0;
	
	public int height = 0;

	public int width = 0;
	
	public int dataLength = 0;
	
	public float[] data;
	
	private Pointer gpuData;
	
	public float[] once;
	
	private boolean hasGPU = false;
	
	public Tensor(int number,int channel,int height,int width) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = new float[this.dataLength];
	}
	
	public Tensor(int number,int channel,int height,int width,boolean hasGPU) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = new float[this.dataLength];
		this.hasGPU = hasGPU;
		if(hasGPU) {
			gpuData = CUDAMemoryManager.getPointer(dataLength);
			JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaDeviceSynchronize();
		}
	}
	
	public Tensor(int number,int channel,int height,int width,float[] data) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = data;
	}
	
	public Tensor(int number,int channel,int height,int width,float[] data,boolean hasGPU) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = data;
		this.hasGPU = hasGPU;
		if(hasGPU) {
			gpuData = CUDAMemoryManager.getPointer(dataLength);
			JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaDeviceSynchronize();
		}
	}
	
	public void copy(int n,float[] dest) {
		if(n < number) {
			System.arraycopy(data, n * channel * height * width, dest, 0, channel * height * width);
		}else {
			throw new RuntimeException("获取数据失败[下标超出长度].");
		}
	}
	
	public int getNumber() {
		return number;
	}

	public void setNumber(int number) {
		this.number = number;
	}

	public int getChannel() {
		return channel;
	}

	public void setChannel(int channel) {
		this.channel = channel;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public int getDataLength() {
		return dataLength;
	}

	public void setDataLength(int dataLength) {
		this.dataLength = dataLength;
	}

	public float[] getData() {
		return data;
	}

	public void setData(float[] data) {
		this.data = data;
		if(hasGPU) {
			this.hostToDevice();
		}
	}

	public float getByIndex(int n,int c,int h,int w) {
		return this.data[n * channel * height * width + c * height * width + h * width + w];
	}
	
	public float[] getByNumber(int n) {
		if(once == null || once.length != channel * height * width) {
			once = new float[channel * height * width];
		}
		System.arraycopy(data, n * channel * height * width, once, 0, channel * height * width);
		return once;
	}
	
	public float[] getByNumberAndChannel(int n,int c) {
		if(once == null || once.length != height * width) {
			once = new float[height * width];
		}
		int start = n * channel * height * width + c * height * width;
		System.arraycopy(data, start, once, 0, height * width);
		return once;
	}
	
	public void clear() {
		for(int i = 0;i<this.dataLength;i++) {
			this.data[i] = 0;
		}
	}
	
	public void clear(int number,int channel,int height,int width) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = new float[this.dataLength];
	}

	public Pointer getGpuData() {
		return gpuData;
	}
	
	public float[] syncHost() {
		JCuda.cudaMemcpy(Pointer.to(data), gpuData, this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		return data;
	}
	
	public void hostToDevice() {
		if(gpuData == null) {
			gpuData = CUDAMemoryManager.getPointer(dataLength);
		}
		JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		JCuda.cudaDeviceSynchronize();
	}
	
	public void showDM() {
		syncHost();
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void clearGPU() {
		checkCUDA(JCuda.cudaMemset(gpuData, 0, this.dataLength * Sizeof.FLOAT));
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
}
