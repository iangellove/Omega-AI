package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.PrintUtils;

/**
 * SoftmaxWithCrossEntropyLayer
 * 
 * @author Administrator
 * 
 * @remark
 * 
 * The softmax function support only cross entropy loss function now.
 * 
 */
public class SoftmaxWithCrossEntropyLayer extends Layer {
	
	private Tensor currentLabel;
	
	public SoftmaxWithCrossEntropyLayer(int inputNum) {
		this.channel = 1;
		this.height = 1;
		this.width = inputNum;
		this.oChannel = channel;
		this.oHeight = height;
		this.oWidth = inputNum;
		this.initParam();
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
		this.output = new Tensor(number, oChannel, oHeight, oWidth);
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		this.diff = new Tensor(number, channel, height, width);
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		float[] dest = new float[channel * height * width];
		
		for(int n = 0;n<this.number;n++) {
			input.copy(n, dest);
			float max = MatrixOperation.max(dest);
			float[] temp = MatrixOperation.subtraction(dest, max);
			temp = MatrixOperation.exp(temp);
			float sum = MatrixOperation.sum(temp);
			for(int i = 0;i<temp.length;i++) {
				this.output.data[n * channel * height * width + i] = temp[i] / sum;
			}
		}
		
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		for(int i = 0;i<this.delta.getDataLength();i++) {
			this.diff.data[i] = this.output.data[i] - this.currentLabel.data[i];
		}

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		
//		System.out.println("====");
//		
//		MatrixOperation.printImage(this.input.maxtir[0]);
		
		this.output();

//		MatrixOperation.printImage(this.output.maxtir[0]);
		
		
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
	}

	public void setCurrentLabel(Tensor currentLabel) {
		this.currentLabel = currentLabel;
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.softmax_cross_entropy;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}

}
