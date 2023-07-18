package com.omega.engine.ad;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
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
import com.omega.engine.ad.op.sign.SubOP;

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
	
	public static void showGraph() {
		for(int i = 0;i<tapes.size();i++) {
			System.out.println(i+":["+tapes.get(i).getOp().getOpType()+"]");
		}
	}
	
	public static void reset() {
		Graph.tapes.clear();
	}
	
	public static void clearGrad() {
		for(Tape tape:Graph.tapes) {
			tape.zeroGrad();
		}
		reset();
	}
	
	public static void add(Tape tape) {
		Graph.tapes.add(tape);
	}
	
	public static Tensor OP(OPType op,Tensor self,Tensor other) {
		
		switch (op) {
		case add:
			return AddOP.getInstance().forward(self, other);
		case subtraction:
			return SubOP.getInstance().forward(self, other);
		case multiplication:
			return MulOP.getInstance().forward(self, other);
		case division:
			return DivOP.getInstance().forward(self, other);
		default:
			break;	
		}
		
		return null;
	}
	
	public static Tensor OP(OPType op,Tensor self,float other) {
		
		switch (op) {
		case add:
			return AddOP.getInstance().forward(self, other);
		case subtraction:
			return SubOP.getInstance().forward(self, other);
		case multiplication:
			return MulOP.getInstance().forward(self, other);
		case division:
			return DivOP.getInstance().forward(self, other);
		case scalarDivision:
			return ScalarDivOP.getInstance().forward(self, other);
		default:
			break;	
		}
		
		return null;
	}
	
	public static Tensor OP(OPType op,Tensor self) {
		
		switch (op) {
		case log:
			return LogOP.getInstance().forward(self);
		case sin:
			return SinOP.getInstance().forward(self);
		case exp:
			return ExpOP.getInstance().forward(self);
		case pow:
			return PowOP.getInstance().forward(self);
		default:
			break;	
		}
		
		return null;
	}
	
	public static Tensor OP(OPType op,Tensor self,int[] position) {
		return GetOP.getInstance().forward(self, position);
	}
	
	public static void backward(float[] delta) {
//		float[] preDelta = null;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			if(i == tapes.size() - 1) {
				tape.backward(delta);
			}else {
				tapes.get(i).backward();
			}
//			preDelta = tape.getInputs().get(0).getGrad();
		}
	}
	
	public static void backward() {
		for(int i = tapes.size() - 1;i >= 0;i--) {
			tapes.get(i).backward();
		}
	}
	
	public static void main(String[] args) {
		
		/**
		 * f(x,y)=ln(x)+x*y−sin(y)
		 */
//		formula1();
		
		/**
		 * sigmoid: 1 / 1 + exp(-x)
		 */
		sigmoid();
		
		
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
	
	public static void sigmoid() {
		
		int number = 10;
		int channel  = 5;
		int height = 5;
		int width = 5;
		int length = number * channel * height * width;
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 0.6f));
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 1f));
		
		x.setRequiresGrad(true);

		Tensor v1 = x.get(new int[] {1,0,2}).mul(-1).exp().add(1).scalarDiv(1);
		
		Tensor v2 = x.get(new int[] {1,2,2});
		
		Tensor v3 = x.get(new int[] {1,4,1}).mul(-1).exp().add(1).scalarDiv(1);
		
		Tensor v4 = y.get(new int[] {1,4,1}).sub(v3).pow();

		Graph.showGraph();
		
		Graph.backward();
		
		System.out.println("z1:"+JsonUtils.toJson(v1.data));
		System.out.println("z2:"+JsonUtils.toJson(v2.data));
		System.out.println("z3:"+JsonUtils.toJson(v3.data));
		System.out.println("z4:"+JsonUtils.toJson(v4.data));
		System.out.println("dx:"+JsonUtils.toJson(x.getGrad()));
		
		PrintUtils.printImage(x.getGrad(), x.number, x.channel, x.height, x.width);
		
		System.out.println(1 / (1 + Math.exp(-0.6f)));
		
	}
	
}
