package com.omega.common.data;

import java.io.Serializable;

import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.op.OPType;
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
	
	public int onceSize;
	
	private boolean hasGPU = false;
	
	private boolean requiresGrad = false;
	
	private float[] grad;
	
	public Tensor copy() {
		float[] dest = new float[dataLength];
		System.arraycopy(data, 0, dest, 0, dataLength);
		Tensor dis = new Tensor(number, channel, height, width, dest, hasGPU);
		return dis;
	}
	
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
		this.setHasGPU(hasGPU);
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
		this.setHasGPU(hasGPU);
		if(hasGPU) {
			gpuData = CUDAMemoryManager.getPointer(dataLength);
			JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaDeviceSynchronize();
		}
	}
	
	public Tensor(int number,int channel,int height,int width,int val,boolean hasGPU) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.dataLength = number * channel * height * width;
		this.data = MatrixUtils.val(this.dataLength, val);
		this.setHasGPU(hasGPU);
		if(hasGPU) {
			hostToDevice();
		}
	}
	
	public void copy(int n,float[] dest) {
		if(n < number) {
			System.arraycopy(data, n * channel * height * width, dest, 0, channel * height * width);
		}else {
			throw new RuntimeException("获取数据失败[下标超出长度].");
		}
	}
	
	public int[] shape() {
		return new int[] {this.number,this.channel,this.height,this.width};
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
		return number * channel * height * width;
	}
	
	public void setDataLength(int dataLength) {
		this.dataLength = dataLength;
	}
	
	public int getOnceSize() {
		return channel * height * width;
	}
	
	public float[] getData() {
		return data;
	}

	public void setData(float[] data) {
		this.data = data;
		if(isHasGPU()) {
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
	
	public void setByNumber(int n,float[] x) {
		System.arraycopy(x, 0, data, n * channel * height * width, channel * height * width);
	}
	
	public void getByNumber(int n,float[] once) {
		if(once == null || once.length != channel * height * width) {
			once = new float[channel * height * width];
		}
		System.arraycopy(data, n * channel * height * width, once, 0, channel * height * width);
	}
	
	public float[] getByNumberAndChannel(int n,int c) {
		if(once == null || once.length != height * width) {
			once = new float[height * width];
		}
		int start = n * channel * height * width + c * height * width;
		System.arraycopy(data, start, once, 0, height * width);
		return once;
	}
	
	public void setByNumberAndChannel(int n,int c,float[] x) {
		System.arraycopy(x, 0, data, n * channel * height * width + c * height * width, height * width);
	}
	
	public void getByNumberAndChannel(int n,int c,float[] once) {
		if(once == null || once.length != height * width) {
			once = new float[height * width];
		}
		int start = n * channel * height * width + c * height * width;
		System.arraycopy(data, start, once, 0, height * width);
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
	
	public void setGpuData(Pointer gpuData) {
		this.gpuData = gpuData;
	}
	
	public float[] syncHost() {
		JCuda.cudaMemcpy(Pointer.to(data), gpuData, this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);
		return data;
	}
	
	public void hostToDevice() {
		if(hasGPU) {
			if(gpuData == null) {
				gpuData = CUDAMemoryManager.getPointer(dataLength);
			}
			JCuda.cudaMemcpy(gpuData, Pointer.to(data), this.dataLength * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyHostToDevice);
			JCuda.cudaDeviceSynchronize();
		}
	}
	
	public void freeGPU() {
		if(gpuData != null) {
			JCuda.cudaFree(gpuData);
			gpuData = CUDAMemoryManager.getPointer(dataLength);
		}
	}
	
	public void showDM() {
		syncHost();
	    System.out.println(JsonUtils.toJson(data));
	}
	
	public void showDMByNumber(int number) {
		syncHost();
	    System.out.println(JsonUtils.toJson(this.getByNumber(number)));
	}
	
	public void showDM(int index) {
		syncHost();
	    System.out.println(data[index]);
	}
	
	public boolean checkDM() {
		for(float val:syncHost()) {
			if(val > 0) {
				return true;
			}
		}
	    return false;
	}
	
	public void clearGPU() {
		checkCUDA(JCuda.cudaMemset(gpuData, 0, this.dataLength * Sizeof.FLOAT));
	}
	
	public void valueGPU(int val) {
		checkCUDA(JCuda.cudaMemset(gpuData, val, this.dataLength * Sizeof.FLOAT));
	}
	
	public void checkCUDA(int code) {
		if(code != cudaError.cudaSuccess) {
			System.err.println("Error code "+code+":"+cudaError.stringFor(code));
		}
	}
	
	public boolean isHasGPU() {
		return hasGPU;
	}

	public void setHasGPU(boolean hasGPU) {
		this.hasGPU = hasGPU;
	}

	public boolean isRequiresGrad() {
		return requiresGrad;
	}

	public void setRequiresGrad(boolean requiresGrad) {
		this.requiresGrad = requiresGrad;
	}

	public float[] getGrad() {
		return grad;
	}

	public void setGrad(float[] grad) {
		this.grad = grad;
	}
	
	public void setGrad(float[] grad,int[] position) {
		
		if(this.grad == null) {
			this.grad = new float[this.dataLength];
		}
		
		int dims = position[0];
		int start = position[1];
		int count = position[2];

		switch (dims) {
		case 0:
			setGradByNumber(grad, start, count);
			break;
		case 1:
			setGradByChannel(grad, start, count);
			break;
		default:
			break;
		}
		
	}
	
	public void zeroGrad() {
		if(this.grad!=null) {
			this.grad = new float[this.grad.length];
		}
	}

	/**
	 * tensor基础操作
	 * @return
	 */
	public Tensor add(Tensor y) {
		return Graph.OP(OPType.add, this, y);
	}
	
	public Tensor add(float y) {
		return Graph.OP(OPType.add, this, y);
	}
	
	public Tensor sub(Tensor y) {
		return Graph.OP(OPType.subtraction, this, y);
	}
	
	public Tensor sub(float y) {
		return Graph.OP(OPType.subtraction, this, y);
	}
	
	public Tensor mul(Tensor y) {
		return Graph.OP(OPType.multiplication, this, y);
	}
	
	public Tensor mul(float scalar) {
		return Graph.OP(OPType.multiplication, this, scalar);
	}
	
	public Tensor div(Tensor y) {
		return Graph.OP(OPType.division, this, y);
	}
	
	public Tensor div(float scalar) {
		return Graph.OP(OPType.division, this, scalar);
	}
	
	public Tensor scalarDiv(float scalar) {
		return Graph.OP(OPType.scalarDivision, this, scalar);
	}
	
	public Tensor log() {
		return Graph.OP(OPType.log, this);
	}
	
	public Tensor pow() {
		return Graph.OP(OPType.pow, this);
	}
	
	public Tensor sin() {
		return Graph.OP(OPType.sin, this);
	}
	
	public Tensor exp() {
		return Graph.OP(OPType.exp, this);
	}
	
	public Tensor get(int[] position) {
		return Graph.OP(OPType.get, this, position);
	}
	
	public void setGradByNumber(float[] data,int start,int count) {
		assert number >= (start + count - 1);
		System.arraycopy(data, 0, this.grad, start * channel * height * width, data.length);
	}
	
	public void setGradByChannel(float[] data,int start,int count) {
		assert channel >= (start + count - 1);
		int size = height * width;
		for(int n = 0;n<number;n++) {
			int startIndex = n * channel * size + start * size;
			System.arraycopy(data, n * count * size, this.grad, startIndex, count * size);
		}
	}
	
}
