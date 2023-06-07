package com.omega.engine.ad;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.sign.AddOP;
import com.omega.engine.ad.op.sign.LogOP;
import com.omega.engine.ad.op.sign.MulOP;
import com.omega.engine.ad.op.sign.SinOP;
import com.omega.engine.ad.op.sign.SubOP;

/**
 * 计算图
 * @author Administrator
 *
 */
public class Graph{

	private static List<Tape> tapes = new ArrayList<Tape>();
	
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
		default:
			break;	
		}
		
		return null;
	}
	
	public static Tensor OP(OPType op,Tensor self) {
		
		switch (op) {
		case log:
			return LogOP.getInstance().forward(self, null);
		case sin:
			return SinOP.getInstance().forward(self, null);
		default:
			break;	
		}
		
		return null;
	}
	
	public static void backward(float[] delta) {
		float[] preDelta = null;
		for(int i = tapes.size() - 1;i >= 0;i--) {
			Tape tape = tapes.get(i);
			if(i == tapes.size() - 1) {
				tape.backward(delta);
			}else {
				tape.backward(preDelta);
			}
			preDelta = tape.getInputs().get(0).getGrad();
		}
	}
	
	public static void backward() {
		for(int i = tapes.size() - 1;i >= 0;i--) {
			tapes.get(i).backward();
		}
	}
	
	public static void main(String[] args) {
		
		int number = 1;
		int channel  = 1;
		int height = 1;
		int width = 1;
		int length = number * channel * height * width;
		
		Tensor x = new Tensor(number, channel, height, width, MatrixUtils.val(length, 2.0f));
		
		Tensor y = new Tensor(number, channel, height, width, MatrixUtils.val(length, 5.0f));
		
		x.setRequiresGrad(true);
		y.setRequiresGrad(true);
		
//		Tensor v1 = x.log();
//		
//		Tensor v2 = x.mul(y); //10
//		
//		Tensor v3 = y.sin();
//		
//		Tensor v4 = v1.add(v2);
//		
//		Tensor v5 = v4.sub(v3);
		
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
	
}
