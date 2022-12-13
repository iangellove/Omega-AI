package com.omega.engine.nn.data;

import com.omega.common.data.Tensors;

/**
 * data set
 * @author Administrator
 *
 */
public class DataSet extends BaseData{
	
	public int inputSize = 0;
	
	public DataSet(int number,int channel,int height,int width,int labelSize,float[] dataInput,float[] dataLabel,String[] labels,String[] labelSet) {
		this.number = number;
		this.channel = channel;
		this.height = height;
		this.width = width;
		this.inputSize = number * channel * height * width;
		this.labelSize = labelSize;
		this.labels = labels;
		this.labelSet = labelSet;
		this.input = Tensors.tensor(number, channel, height, width, dataInput);
		this.label = Tensors.tensor(number, 1, 1, labelSet.length, dataLabel);
	}
	
}
