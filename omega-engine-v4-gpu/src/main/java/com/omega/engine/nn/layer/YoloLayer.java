package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.BaseKernel;
import com.omega.engine.nn.layer.active.gpu.SigmodKernel;
import com.omega.engine.nn.network.Yolo;

/**
 * yolo layer
 * 负责对yolov3模型输出层的logistic变换,
 * yolov3的输出层预测的宽高(w,h)不进行变换
 * @author Administrator
 *
 */
public class YoloLayer extends Layer {
	
	private SigmodKernel kernel;
	
	private BaseKernel baseKernel;
	
	private Layer preLayer;
	
	public int class_number = 1;
	
	public int bbox_num = 3;
	
	public int total = 6;
	
	public int[] mask;
	
	public float[] anchors;
	
	public int maxBox = 90;
	
	public float ignoreThresh = 0.7f;
	
	public float truthThresh = 1.0f;
	
	public int outputs = 0;
	
	public YoloLayer(int class_number,int bbox_num,int[] mask,float[] anchors,int maxBox,int total,float ignoreThresh,float truthThresh) {
		this.class_number = class_number;
		this.bbox_num = bbox_num;
		this.mask = mask;
		this.anchors = anchors;
		this.maxBox = maxBox;
		this.total = total;
		this.ignoreThresh = ignoreThresh;
		this.truthThresh = truthThresh;
		this.isOutput = true;
	}
	
	@Override
	public void init() {
		// TODO Auto-generated method stub

		this.number = this.network.number;
		
		if(kernel == null) {
			kernel = new SigmodKernel();
		}
		
		if(baseKernel == null) {
			baseKernel = new BaseKernel();
		}

		if(this.preLayer == null) {
			this.preLayer = this.network.getPreLayer(this.index);
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		
		if(output == null || number != output.number) {
			output = new Tensor(number, oChannel, oHeight, oWidth, true);
		}
		
		this.outputs = bbox_num*(class_number + 4 + 1) * this.height*this.width;

	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
//		if(this.diff == null) {
//			this.diff = this.network.getDelta(this.index);
//		}
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}
	
	public int entryIndex(int batch,int location,int entry) {
		int n = location / (width * height);
	    int loc = location % (width * height);
	    return batch * outputs + n * width * height * (4 + class_number + 1) + entry * width * height + loc;
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		baseKernel.copy_gpu(input, output, input.dataLength, 1, 1);
//		System.out.println("bbox_num:"+bbox_num);
		for(int b = 0;b<this.input.number;b++) {
			for(int n = 0;n<bbox_num;n++) {
				int index = entryIndex(b, n * width * height, 0);
//				int index = b * this.outputs + n * input.width * input.height * (4+1+class_number);
				kernel.forward(input, output, index, 2 * input.width * input.height);
//				index = b * this.outputs + n * input.width * input.height * (4+1+class_number) + 4 * input.width * input.height;
				int index2 = entryIndex(b, n * width * height, 4);
				kernel.forward(input, output, index2, (1+class_number) * input.width * input.height);
			}
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
		this.diff = this.delta;
//		System.out.println(this.delta);
//		baseKernel.copy_gpu(this.delta, this.diff, this.delta.dataLength, 1, 1);
//		for(int b = 0;b<this.delta.number;b++) {
//			for(int n = 0;n<bbox_num;n++) {
//				int index = entryIndex(b, n * width * height, 0);
////				int index = b * this.outputs + n * input.width * input.height * (4+1+class_number);
//				kernel.forward(input, output, index, 2 * input.width * input.height);
////				index = b * this.outputs + n * input.width * input.height * (4+1+class_number) + 4 * input.width * input.height;
//				int index2 = entryIndex(b, n * width * height, 4);
//				kernel.forward(input, output, index2, (1+class_number) * input.width * input.height);
//			}
//		}
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
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
//		this.setDelta();
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
		return LayerType.yolo;
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
