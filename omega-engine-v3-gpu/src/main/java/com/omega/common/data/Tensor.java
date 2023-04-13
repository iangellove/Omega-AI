package com.omega.common.data;

import java.io.Serializable;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.gpu.CUDAMemoryManager;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
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
	
	private Pointer gpuData;
	
	public float[] once;
	
	private boolean hasGPU = false;
	
	public boolean requiresGrad = false;
	
	public int dataLength = 0;
	
	public float[] data;
	
	public float[] grad;
	
	public Tensor[] dependency;
	
	private OPType opType;
	
	
	public Tensor(int number,int channel,int height,int width) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = new float[this.dataLength];
		if(requiresGrad) {
			grad = new float[dataLength];
			dependency = new Tensor[2];
		}
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
		if(requiresGrad) {
			grad = new float[dataLength];
			dependency = new Tensor[2];
		}
	}
	
	public Tensor(int number,int channel,int height,int width,float[] data) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = data;
		if(requiresGrad) {
			grad = new float[dataLength];
			dependency = new Tensor[2];
		}
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
		if(requiresGrad) {
			grad = new float[dataLength];
			dependency = new Tensor[2];
		}
	}
	
	public void clearGrad() {
		grad = new float[grad.length];
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
	
	public void syncHost() {
		JCuda.cudaMemcpy(Pointer.to(data), gpuData, this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
	}
	
	public void hostToDevice() {
		if(gpuData == null) {
			gpuData = CUDAMemoryManager.getPointer(dataLength);
		}
		JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
		JCuda.cudaDeviceSynchronize();
	}
	
	public void showDM() {
		if(hasGPU) {
			syncHost();
		}
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void showDM2() {
		syncHost();
	    for(float t:data) {
	    	System.out.println(t);
	    }
	}
	
	public void clearGPU() {
		JCuda.cudaMemset(gpuData, 0, data.length * Sizeof.FLOAT);
	}
	
	public void backward(float[] grad) {
		
		if(requiresGrad) {
			
			if(grad == null) {
				grad = MatrixUtils.one(this.grad.length);
			}
			
			this.grad = MatrixOperation.add(this.grad, grad);
			
			dependency[0].backward(dependency[0].gradFunction(this.grad, dependency[1]));
			dependency[1].backward(dependency[1].gradFunction(this.grad, dependency[0]));
		}
		
	}
	
	public float[] gradFunction(float[] grad,Tensor dep) {
		
		switch (this.opType) {
		case add:
			return grad;
		case sub:
			return grad;
		case mul:
			return MatrixOperation.multiplication(grad, dep.data);
		case div:
			return MatrixOperation.division(grad, dep.data);
		case dot:
			
			break;
		default:
			return null;
		}
		
		return null;
	}
	
	public Tensor add(Tensor other) {
		this.opType = OPType.add;
		float[] val = MatrixOperation.add(this.data, other.data);
		
		return new Tensor(number, channel, height, width, val);
	}
	
}
