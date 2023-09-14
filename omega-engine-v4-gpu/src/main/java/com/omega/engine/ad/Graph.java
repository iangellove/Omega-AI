package com.omega.engine.ad;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.data.GetOP;
import com.omega.engine.ad.op.functions.ClampOP;
import com.omega.engine.ad.op.functions.ExpOP;
import com.omega.engine.ad.op.functions.LogOP;
import com.omega.engine.ad.op.functions.PowOP;
import com.omega.engine.ad.op.functions.SinOP;
import com.omega.engine.ad.op.functions.SumOP;
import com.omega.engine.ad.op.sign.AddOP;
import com.omega.engine.ad.op.sign.DivOP;
import com.omega.engine.ad.op.sign.MulOP;
import com.omega.engine.ad.op.sign.ScalarDivOP;
import com.omega.engine.ad.op.sign.ScalarSubOP;
import com.omega.engine.ad.op.sign.SubOP;
import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;

/**
 * 计算图
 * @author Administrator
 *
 */
public class Graph{
	
	/**
	 * 计算图map
	 */
	private List<Tape> tapes = new ArrayList<Tape>();
	
	public int tapeIndex = 0;
	
	private boolean lock = false;
	
	public void start() {
		tapeIndex = 0;
	}
	
	public void lock() {
		lock = true;
	}
	
	public void unlock() {
		lock = false;
	}
	
	public void showGraph() {
		for(int i = 0;i<tapes.size();i++) {
			System.out.println(i+":["+tapes.get(i).getOp().getOpType()+"]");
		}
	}
	
	public void reset() {
		this.tapes.clear();
	}
	
	public void clearGrad() {
		for(int i = 0;i<tapes.size();i++) {
			this.tapes.get(i).zeroGrad();
		}
//		reset();
	}
	
	public void add(Tape tape) {
		this.tapes.add(tape);
	}
	
	public Tape getTape(OP op,Tensor self,Tensor other,float scalar,float constant,int[] position) {
		Tape tape = null;
		if(!lock) {
			tape = new Tape(op, self, other, scalar, constant, position, this);
//			System.out.println(tape.getOp().getOpType()+":"+tape.isSub());
			checkSubTape(self, other);
			
			this.add(tape);
		}else {
			tape = tapes.get(tapeIndex);
			if(tape.getOp().getOpType().equals(OPType.sum)) {
				tape.getOutput().fill(0.0f);
			}
			tapeIndex++;
		}
		return tape;
	}
	
	public void checkSubTape(Tensor a,Tensor b) {
		for(Tape tape:tapes) {
			if(!tape.isSub() && (tape.getX() == a || tape.getY() == a || tape.getX() == b || tape.getY() == b ||
					tape.getOutput() == a || tape.getOutput() == b)) {
//				System.out.println(this.toString()+":"+tape.getOp().getOpType().toString());
				tape.setSub(true);
			}
		}
	}
	
	public Tensor OP(OPType opType,Tensor self,Tensor other) {
		OP op = null;
		switch (opType) {
		case add:
			op = AddOP.getInstance();
			break;
		case subtraction:
			op = SubOP.getInstance();
			break;
		case multiplication:
			op = MulOP.getInstance();
			break;
		case division:
			op = DivOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = this.getTape(op, self, other, 0, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,float other) {
		
		OP op = null;
		
		switch (opType) {
		case add:
			op = AddOP.getInstance();
			break;
		case subtraction:
			op = SubOP.getInstance();
			break;
		case scalarSubtraction:
			op = ScalarSubOP.getInstance();
			break;
		case multiplication:
			op = MulOP.getInstance();
			break;
		case division:
			op = DivOP.getInstance();
			break;
		case scalarDivision:
			op = ScalarDivOP.getInstance();
			break;
		case pow:
			op = PowOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, other, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,float constant1,float constant2) {
		
		OP op = null;
		
		switch (opType) {
		case clamp:
			op = ClampOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, constant1, constant2, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self) {
		
		OP op = null;
		
		switch (opType) {
		case log:
			op = LogOP.getInstance();
			break;
		case sin:
			op = SinOP.getInstance();
			break;
		case exp:
			op = ExpOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = this.getTape(op, self, null, 0, 0, null);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public Tensor OP(OPType opType,Tensor self,int[] position) {
		OP op = null;
		
		switch (opType) {
		case get:
			op = GetOP.getInstance();
			break;
		case sum:
			op = SumOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, 0, 0, position);
		Tensor output = tape.forward();
		output.setG(this);
		return output;
	}
	
	public void backward(Tensor delta) {
//		float[] preDelta = null;
		this.lock = true;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			if(i == tapes.size() - 1) {
				tape.backward(delta);
			}else {
				tape.backward();
			}
//			preDelta = tape.getInputs().get(0).getGrad();
		}
		this.tapeIndex = 0;
	}
	
	public void backward() {
		this.lock = true;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			/**
			 * 初始化最后一代的grad
			 */
//			System.out.println(tape.getOp().getOpType().toString()+":"+tape.isSub());
			if(!tape.isSub()) {
				tape.getOutput().getGrad().fill(1.0f);
			}
			tape.backward();
		}
		this.tapeIndex = 0;
	}
	
	public void formula1(){
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 1;
		int length = number * channel * height * width;
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 2.0f));
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 5.0f));
		
		x.setRequiresGrad(true);
		y.setRequiresGrad(true);
		
		for(int i = 0;i<10;i++) {
			
			this.clearGrad();
			
			/**
			 * f(x,y)=ln(x)+x*y−sin(y)
			 */
			Tensor v5 = x.log().add(x.mul(y)).sub(y.sin());
			
			this.backward();
			
//			System.out.println(JsonUtils.toJson(Graph.tapes));
			
			System.out.println("z:"+JsonUtils.toJson(v5.data));
			
			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
			
			System.out.println("dy:"+JsonUtils.toJson(y.getGrad()));

		}
	}
	
	public void sigmoid_gpu(Tensor x,Tensor y) {
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		long start = System.nanoTime();
		
		Tensor v1 = x.get(1, 0, 2).mul(-1).exp().add(1).scalarDiv(1);
		
		Tensor v2 = x.get(1, 2, 2);
		
		Tensor v3 = y.get(1, 4, 2).sub(x.get(1, 4, 2).mul(-1).exp().add(1).scalarDiv(1)).pow(2);
		
		Tensor z = v1.add(v2).add(v3);
		
//		Graph.showGraph();
		
		this.backward();

		z.syncHost();
		
//		System.out.println("z:"+JsonUtils.toJson(z.data));
		x.getGrad().syncHost();
//		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static void get_gpu() {
		
		int number = 64;
		int channel  = 128;
		int height = 32;
		int width = 32;
		int length = number * channel * height * width;
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
//		x.showDM();
		
		Tensor v1 = x.get(1, 1, 10).pow(2.0f);
		
		Tensor v2 = x.get(1, 14, 10);
		
		graph.showGraph();
		
		graph.backward();
		
		v1.syncHost();
		
		v2.syncHost();
		
		x.getGrad().syncHost();
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//		System.out.println("z1:"+JsonUtils.toJson(v1.data));
		
//		PrintUtils.printImage(v1);
//		
//		System.out.println("*********************************************");
//		
//		PrintUtils.printImage(v2);
//
//		System.out.println("++++++++++++++++++++++++++++++++++++++++++++");
//		
//		PrintUtils.printImage(x.getGrad());

	}
	
	public static void pow_gpu() {
		
		int number = 2;
		int channel  = 3;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		Tensor v1 = x.pow(3);
		
		graph.showGraph();
		
		graph.backward();
		
		v1.syncHost();
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");

	}
	
	public static void show() {
		
		int n = 10;
		int c  = 5;
		int h = 5;
		int w = 5;
		int length = n * c * h * w;
		int count = 2;
		int start = 1;
		
		Tensor x = new Tensor(n, c, h, w, MatrixUtils.order(length, 0, 1));
		
		Tensor y = new Tensor(x.number, count, x.height, x.width, x.isHasGPU());
		
		for(int i = 0;i<y.dataLength;i++) {
			int bc = y.dataLength / n / h / w;
			int size = bc * h * w;
	    	int tn = i / size;
			int tc = (i / h / w) % bc + start;
			int th = (i / w) % h;
			int tw = i % h;
			int index = tn * c * h * w + tc * h * w + th * w + tw;
	    	y.data[i] = x.data[index];
		}
		
		PrintUtils.printImage(y);
	}
	
	public static void yolov3_loss() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		
		int classNum = 1;
		
		int bboxNum = 3;
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true);
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 1f), true);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		long start = System.nanoTime();
		
		Tensor xy1 = BCELoss(sigmoid(x.get(1, 0, 2)), y.get(1, 0, 2));
		
		Tensor wh1 = MSELoss(x.get(1, 2, 2), y.get(1, 2, 2));
		
		Tensor cc1 = BCELoss(sigmoid(x.get(1, 4, 2)), y.get(1, 4, 2));
		
		Tensor xy2 = BCELoss(sigmoid(x.get(1, 6, 2)), y.get(1, 6, 2));
		
		Tensor wh2 = MSELoss(x.get(1, 8, 2), y.get(1, 8, 2));
		 
		Tensor cc2 = BCELoss(sigmoid(x.get(1, 10, 2)), y.get(1, 10, 2));
		
		Tensor z = xy1.add(wh1).add(cc1).add(xy2).add(wh2).add(cc2);
		
//		Graph.showGraph();
		
		graph.backward();

		z.syncHost();
		
		System.out.println("z:"+JsonUtils.toJson(z.data));
		x.getGrad().syncHost();
		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
	
		PrintUtils.printImage(z);
		
		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static Tensor sigmoid(Tensor x) {
		return x.mul(-1).exp().add(1).scalarDiv(1);
	}
	
	public static Tensor MSELoss(Tensor pred,Tensor target) {
		// y = (pred - sub)^2
		return pred.sub(target).pow(2);
	}
	
	public static Tensor BCELoss(Tensor pred,Tensor target) {
		// y = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
		return target.mul(-1).mul(pred.log()).sub(target.scalarSub(1.0f).mul(pred.scalarSub(1.0f).log()));
	}
	
	public static void multiLabelSoftMarginLoss() {
	
		int number = 2;
		int channel  = 1;
		int height = 1;
		int width = 4;
		int length = number * channel * height * width;
		int C = channel * height * width;

		float[] xa = new float[] {0.2f,0.5f,0,0,0.1f,0.5f,0,0.8f};
		
		float[] ya = new float[] {1,1,0,0,0,1,0,1};
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, xa, true);
		
		Tensor y = new Tensor(number, channel, height, width, ya, true);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		/**
		 * -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
		 */
		for(int i = 0;i<20;i++) {

			long start = System.nanoTime();
			
			Tensor x0 = sigmoid(x).log();
			
			Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(y.scalarSub(1.0f));
			
			Tensor loss = y.mul(x0).add(x1).mul(-1.0f);
			
			loss = loss.sum(1).div(C).sum(0).div(x.number);
			
			graph.clearGrad();
			
			graph.backward();
			
			loss.syncHost();
			
			System.out.println("loss:"+JsonUtils.toJson(loss.data));
			
			x.getGrad().syncHost();
			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
			
			System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
			PrintUtils.printImage(x.getGrad());
			
		}
		
	}
	
	public static void multiLabelSoftMarginLoss2() {
		
		int number = 64;
		int channel  = 128;
		int height = 32;
		int width = 32;
		int length = number * channel * height * width;
		int C = channel * height * width;

		float[] xa = RandomUtils.gaussianRandom(length, 0.1f);
		
		float[] ya = RandomUtils.gaussianRandom(length, 0.1f);
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, xa, true, graph);
		
		Tensor y = new Tensor(number, channel, height, width, ya, true, graph);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		/**
		 * -(target * logsigmoid(input) + (1 - target) * logsigmoid(-input))
		 */
		for(int i = 0;i<200;i++) {

			long start = System.nanoTime();
			
			graph.start();
			
			Tensor x0 = sigmoid(x).log();
			
			Tensor x1 = sigmoid(x.mul(-1.0f)).log().mul(y.scalarSub(1.0f));
			
			Tensor loss = y.mul(x0).add(x1).mul(-1.0f);
			
			loss = loss.sum(1).div(C).sum(0).div(x.number);
			
			graph.lock = true;
			
			graph.clearGrad();
			
			graph.backward();
			
			loss.syncHost();
			
//			System.out.println("loss:"+JsonUtils.toJson(loss.data));
			
//			x.getGrad().syncHost();
//			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
//			x.getGrad().showDM();
			System.out.println(((System.nanoTime() - start) / 1e6) + "ms.");
		
//			PrintUtils.printImage(x.getGrad());
			
		}
		
	}
	
	public static void sq() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		int C = channel * height * width;
		
		float[] cpx = RandomUtils.gaussianRandom(length, 0.1f);
		
		float[] cpy = RandomUtils.gaussianRandom(length, 0.1f);
		
		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, cpx, true);
		
		Tensor y = new Tensor(number, channel, height, width, cpy, true);
		
		for(int i = 0;i<20;i++) {

			x.data = RandomUtils.gaussianRandom(length, 0.1f);
			
			y.data = RandomUtils.gaussianRandom(length, 0.1f);
			
			sq_back_cpu(x, y);
			
			x.setRequiresGrad(true);
			
			x.hostToDevice();
			y.hostToDevice();
			
			Tensor loss1 = y.sub(x).pow(2.0f).div(2.0f);

			graph.clearGrad();
			
			graph.backward();
			
			x.getGrad().syncHost();
			System.out.println("dx_gpu:"+JsonUtils.toJson(x.getGrad().data));
		}

	}
	
	public static void sq_back_cpu(Tensor x,Tensor y) {
		
		Tensor temp = new Tensor(x.number, x.channel, x.height, x.width, true);
		
		for(int i = 0;i<x.getDataLength();i++) {
			temp.data[i] = x.data[i] - y.data[i];
		}
		System.out.println("dx_cpu:"+JsonUtils.toJson(temp.data));
	}
	
	public static void sum() {
		
		int number = 3;
		int channel  = 18;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;

		Graph graph = new Graph();
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true, graph);
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		Tensor z = x.sum(1);
		
		graph.backward();
		
		z.syncHost();
		
		System.out.println("z:"+JsonUtils.toJson(z.data));
		
		x.getGrad().syncHost();
		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		PrintUtils.printImage(x.getGrad());
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();

			/**
			 * f(x,y)=ln(x)+x*y−sin(y)
			 */
//			formula1();
			
			/**
			 * sigmoid: 1 / 1 + exp(-x)
			 */
//			sigmoid();
			
//			get_gpu();
			
//			int number = 64;
//			int channel  = 125;
//			int height = 32;
//			int width = 32;
//			int length = number * channel * height * width;
//			
//			Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f), true);
//			
//			Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 1f), true);
//			
//			sigmoid_gpu(x, y);
//			
//			x.data = MatrixUtils.val(length, 0.35f);
//			
//			y.data = MatrixUtils.val(length, 2f);
//			
//			Graph.clearGrad();
//			
//			sigmoid_gpu(x, y);
			
//			show();
			
//			pow_gpu();
			
			multiLabelSoftMarginLoss2();
			
//			sq();
			
//			sum();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
			
		}
		
	}
	
	
}
