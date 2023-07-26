package com.omega.engine.ad;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.data.GetOP;
import com.omega.engine.ad.op.functions.ExpOP;
import com.omega.engine.ad.op.functions.LogOP;
import com.omega.engine.ad.op.functions.PowOP;
import com.omega.engine.ad.op.functions.SinOP;
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
	private static List<Tape> tapes = new ArrayList<Tape>();
	
	private static int tapeIndex = 0;
	
	private static boolean lock = false;
	
	public static void showGraph() {
		for(int i = 0;i<tapes.size();i++) {
			System.out.println(i+":["+tapes.get(i).getOp().getOpType()+"]");
		}
	}
	
	public static void reset() {
		Graph.tapes.clear();
	}
	
	public static void clearGrad() {
		for(int i = 0;i<tapes.size();i++) {
			Graph.tapes.get(i).zeroGrad();;
		}
//		reset();
	}
	
	public static void add(Tape tape) {
		Graph.tapes.add(tape);
	}
	
	public static Tape getTape(OP op,Tensor self,Tensor other,float scalar,int[] position) {
		Tape tape = null;
		if(!lock) {
			tape = new Tape(op, self, other, scalar, position);
			if(self.getTape() != null) {
				self.getTape().setSub(true);
			}
			self.setTape(tape);
			if(other != null) {
				if(other.getTape() != null) {
					other.getTape().setSub(true);
				}
				other.setTape(tape);
			}
			Graph.add(tape);
		}else {
			tape = tapes.get(tapeIndex);
			tapeIndex++;
		}
		return tape;
	}
	
	public static Tensor OP(OPType opType,Tensor self,Tensor other) {
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
		
		Tape tape = getTape(op, self, other, 0, null);
		Tensor output = tape.forward();
		output.setTape(tape);
		return output;
	}
	
	public static Tensor OP(OPType opType,Tensor self,float other) {
		
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
		
		Tape tape = getTape(op, self, null, other, null);
		Tensor output = tape.forward();
		output.setTape(tape);
		return output;
	}
	
	public static Tensor OP(OPType opType,Tensor self) {
		
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
		
		Tape tape = getTape(op, self, null, 0, null);
		Tensor output = tape.forward();
		output.setTape(tape);
		return output;
	}
	
	public static Tensor OP(OPType opType,Tensor self,int[] position) {
		OP op = null;
		
		switch (opType) {
		case get:
			op = GetOP.getInstance();
			break;
		default:
			break;	
		}
		
		if(op == null) {
			throw new RuntimeException("the op is not support.");
		}
		
		Tape tape = getTape(op, self, null, 0, position);
		Tensor output = tape.forward();
		output.setTape(tape);
		return output;
	}
	
	public static void backward(Tensor delta) {
//		float[] preDelta = null;
		Graph.lock = true;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			if(i == tapes.size() - 1) {
				tape.backward(delta);
			}else {
				tape.backward();
			}
//			preDelta = tape.getInputs().get(0).getGrad();
		}
		Graph.tapeIndex = 0;
	}
	
	public static void backward() {
		Graph.lock = true;
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
		Graph.tapeIndex = 0;
	}
	
	public static void formula1(){
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
			
			Graph.clearGrad();
			
			/**
			 * f(x,y)=ln(x)+x*y−sin(y)
			 */
			Tensor v5 = x.log().add(x.mul(y)).sub(y.sin());
			
			Graph.backward();
			
//			System.out.println(JsonUtils.toJson(Graph.tapes));
			
			System.out.println("z:"+JsonUtils.toJson(v5.data));
			
			System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
			
			System.out.println("dy:"+JsonUtils.toJson(y.getGrad()));

		}
	}
	
	public static void sigmoid_gpu(Tensor x,Tensor y) {
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		y.hostToDevice();
		
		long start = System.nanoTime();
		
		Tensor v1 = x.get(1, 0, 2).mul(-1).exp().add(1).scalarDiv(1);
		
		Tensor v2 = x.get(1, 2, 2);
		
		Tensor v3 = y.get(1, 4, 2).sub(x.get(1, 4, 2).mul(-1).exp().add(1).scalarDiv(1)).pow(2);
		
		Tensor z = v1.add(v2).add(v3);
		
//		Graph.showGraph();
		
		Graph.backward();

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
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
//		x.showDM();
		
		Tensor v1 = x.get(1, 1, 10).pow(2.0f);
		
		Tensor v2 = x.get(1, 14, 10);
		
		Graph.showGraph();
		
		Graph.backward();
		
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
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.order(length, 0, 1), true);
		
		long start = System.nanoTime();
		
		x.setRequiresGrad(true);
		
		x.hostToDevice();
		
		Tensor v1 = x.pow(3);
		
		Graph.showGraph();
		
		Graph.backward();
		
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
		
		Graph.backward();

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
			
			yolov3_loss();
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
			
		}
		
	}
	
	
}
