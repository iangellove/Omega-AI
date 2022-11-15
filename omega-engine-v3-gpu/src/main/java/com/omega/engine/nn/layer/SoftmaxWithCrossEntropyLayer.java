package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.gpu.SoftmaxKernel;

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
	
	private SoftmaxKernel kernel;
	
	private Tensor cl;
	
	private Tensor x;
	
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
		if(this.output == null || this.number != this.output.number) {
			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}
		if(kernel == null) {
			kernel = new SoftmaxKernel();
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
		if(this.diff == null || this.number != this.diff.number) {
			this.diff = new Tensor(number, channel, height, width, true);
		}
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		
//		System.out.println("softmax-in:");
		
		if(x == null) {
			x = new Tensor(number, channel, height, width, input.data, true);
		}else {
			x.setData(input.data);
		}
		
		kernel.forward(x, output);
//		
//		float[] dest = new float[channel * height * width];
//		
//		for(int n = 0;n<this.number;n++) {
//			input.copy(n, dest);
//			float max = MatrixOperation.max(dest);
//			float[] temp = MatrixOperation.subtraction(dest, max);
//			temp = MatrixOperation.exp(temp);
//			float sum = MatrixOperation.sum(temp);
//			for(int i = 0;i<temp.length;i++) {
//				this.output.data[n * channel * height * width + i] = temp[i] / sum;
//			}
//		}
//		
		output.syncHost();
		
//		output.showDM();
		
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
//		
//		for(int i = 0;i<this.delta.getDataLength();i++) {
//			this.diff.data[i] = this.output.data[i] - this.currentLabel.data[i];
//		}
//		
		
		if(cl == null) {
			cl = new Tensor(number, channel, height, width, currentLabel.data, true);
		}else {
			cl.setData(currentLabel.data);
		}
		
		kernel.backward(output, cl, diff);
		
		diff.syncHost();
		
//		diff.showDM();
		
//		diff.showDM();
		
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
		
//		output.syncHost();
//		
//		output.showDM();
		
		
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
	

	@Override
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(inpnut);
		/**
		 * 计算输出
		 */
		this.output();
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}
	}

}
