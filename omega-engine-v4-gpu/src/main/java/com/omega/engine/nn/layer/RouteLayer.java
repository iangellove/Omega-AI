package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.BaseKernel;

/**
 * 路由层
 * @author Administrator
 *
 */
public class RouteLayer extends Layer{
	
	private Layer[] layers;
	
	private BaseKernel kernel;
	
	public RouteLayer(Layer[] layers) {
		this.layers = layers;
		Layer first = layers[0];
		this.oHeight = first.oHeight;
		this.oWidth = first.oWidth;
		for(Layer layer:layers) {
			if(layer.oHeight != this.oHeight || layer.oWidth != this.oWidth) {
				throw new RuntimeException("input size must be all same in the route layer.");
			}
			this.oChannel += layer.oChannel;
		}
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		
		this.number = this.network.number;
		
		if(this.output == null || this.output.number != this.number) {
			this.output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}

		if(kernel == null) {
			kernel = new BaseKernel();
		}
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		if(layers[0].cache_delta == null || layers[0].cache_delta.number != this.number) {
			for(Layer layer:layers) {
				layer.cache_delta = new Tensor(number, layer.oChannel, oHeight, oWidth, true);
			}
		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		int offset = 0;
		for(int l = 0;l<layers.length;l++) {
			Tensor input = layers[l].output;
			for(int n = 0;n<this.number;n++) {
				kernel.copy_gpu(input, this.output, input.getOnceSize(), n * input.getOnceSize(), 1, offset + n * output.getOnceSize(), 1);
			}
			offset += input.getOnceSize();
		}
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		int offset = 0;
		for(int l = 0;l<layers.length;l++) {
			Tensor delta = layers[l].cache_delta;
			for(int n = 0;n<this.number;n++) {
//				kernel.axpy_gpu(this.delta, delta, delta.getOnceSize(), 1, offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize(), 1);
				kernel.copy_gpu(this.delta, delta, delta.getOnceSize(), offset + n * this.delta.getOnceSize(), 1, n * delta.getOnceSize(), 1);
			}
			offset += delta.getOnceSize();
		}
	}
	
	public static void main(String[] args) {
		
		int N = 2;
    	int C = 3;
    	int C2 = 2;
    	int H = 4;
    	int W = 4;
    	
    	int oHeight = H;
		int oWidth = W;
		int oChannel = C + C2;
		
    	float[] x = MatrixUtils.order(N * C * H * W, 1, 1);
    	
    	float[] x2 = MatrixUtils.order(N * C2 * H * W, 1, 1);
    	
    	float[] d = RandomUtils.order(N * oChannel * oHeight * oWidth, 1, 1);

    	Tensor input = new Tensor(N, C, H, W, x, true);
    	
    	Tensor input2 = new Tensor(N, C2, H, W, x2, true);
    	
    	Tensor[] inputs = new Tensor[] {input,input2};
    	
    	Tensor output = new Tensor(N, oChannel, oHeight, oWidth, true);
    	
    	Tensor delta = new Tensor(N, oChannel, oHeight, oWidth, d, true);
    	
    	Tensor diff1 = new Tensor(N, C, H, W, true);
    	
    	Tensor diff2 = new Tensor(N, C2, H, W, true);
    	
    	Tensor[] diffs = new Tensor[] {diff1,diff2};
    	
    	BaseKernel kernel = new BaseKernel();
    	
    	testForward(inputs, output, kernel);
    	
    	output.showDM();
    	
    	testBackward(diffs, delta, kernel);
    	
    	delta.showDM();
    	
    	for(Tensor diff:diffs) {
    		diff.showDM();
    	}
    	
	}
	
	public static void testForward(Tensor[] x,Tensor output,BaseKernel kernel) {

    	int offset = 0;
		for(int l = 0;l<x.length;l++) {
			Tensor input = x[l];
			for(int n = 0;n<output.number;n++) {
				kernel.copy_gpu(input, output, input.getOnceSize(), n * input.getOnceSize(), 1, offset + n * output.getOnceSize(), 1);
			}
			offset += input.getOnceSize();
		}
    	
	}
	
	public static void testBackward(Tensor[] diffs,Tensor delta,BaseKernel kernel) {

		int offset = 0;
		for(int l = 0;l<diffs.length;l++) {
			Tensor diff = diffs[l];
			for(int n = 0;n<delta.number;n++) {
//				kernel.axpy_gpu(delta, diff, diff.getOnceSize(), 1, offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
				kernel.copy_gpu(delta, diff, diff.getOnceSize(), offset + n * delta.getOnceSize(), 1, n * diff.getOnceSize(), 1);
			}
			offset += diff.getOnceSize();
		}
    	
	}
	
	@Override
	public void forward() {
		// TODO Auto-generated method stub

		/**
		 * 参数初始化
		 */
		this.init();

		/**
		 * 计算输出
		 */
		this.output();

	}

	@Override
	public void back() {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.route;
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
