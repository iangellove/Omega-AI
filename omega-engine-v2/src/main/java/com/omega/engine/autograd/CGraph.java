package com.omega.engine.autograd;

import java.io.Serializable;

import com.omega.common.utils.JsonUtils;
import com.omega.engine.autograd.data.Tensor;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.FunctionOP;
import com.omega.engine.autograd.operater.SignOP;
import com.omega.engine.autograd.operater.functions.PowOP;
import com.omega.engine.autograd.operater.signs.AddOP;
import com.omega.engine.autograd.operater.signs.DivisionOP;
import com.omega.engine.autograd.operater.signs.MultiplicationOP;
import com.omega.engine.autograd.operater.signs.SubtractionOP;
import com.omega.engine.autograd.operater.type.FunctionOPType;
import com.omega.engine.autograd.operater.type.SignOPType;

/**
 * ComputationalGraph
 * @author Administrator
 *
 */
public class CGraph implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4853189138813409518L;

	public static Tensor op(Tensor left,Tensor right,SignOPType opType) throws AutogradException {
		SignOP op = null;
		switch (opType) {
		case add:
			op = new AddOP();
			break;
		case subtraction:
			op = new SubtractionOP();
			break;
		case multiplication:
			op = new MultiplicationOP();
			break;
		case division:
			op = new DivisionOP();
			break;
		default:
			break;
		}
		if(op!=null) {
			return op.forward(left, right);
		}
		return null;
	}
	
	public static Tensor op(Tensor left,double e,FunctionOPType opType) throws AutogradException {
		FunctionOP op = null;
		switch (opType) {
		case pow:
			op = new PowOP();
			break;
		case sqrt:
			
			break;
		default:
			break;
		}
		if(op!=null) {
			return op.forward(left, e);
		}
		return null;
	}
	
	public static void main(String[] args) {
		
		double[][][][] xData = new double[][][][] {{{{3}}}};
		
		double[][][][] wData = new double[][][][] {{{{0.1}}}};
		
		double[][][][] bData = new double[][][][] {{{{0.2}}}};
		
		double[][][][] grad = new double[][][][] {{{{0.01}}}};
		
		Tensor x = new Tensor(xData);
		Tensor w = new Tensor(wData);
		Tensor b = new Tensor(bData);
		
		try {
			
//			/**
//			 * y = w * x + b
//			 */
//			Tensor a = CGraph.op(x, w, SignOPType.multiplication);
//			
//			Tensor y = CGraph.op(a, b, SignOPType.add);
//			
//			y.setGrad(grad);
//			
//			y.backward();
//			
//			System.out.println("y:"+JsonUtils.toJson(y.getData()));
//			
//			System.out.println("yGrad:"+JsonUtils.toJson(y.getGrad()));
//			
//			System.out.println("aGrad:"+JsonUtils.toJson(a.getGrad()));
//			
//			System.out.println("wGrad:"+JsonUtils.toJson(w.getGrad()));
//			
//			System.out.println("bGrad:"+JsonUtils.toJson(b.getGrad()));
//			
//			System.out.println("xGrad:"+JsonUtils.toJson(x.getGrad()));
//			
			
			/**
			 * y = x ^ 2 * 3 + 5
			 */

			Tensor three = new Tensor(new double[][][][] {{{{3}}}});
			
			Tensor five = new Tensor(new double[][][][] {{{{5}}}});
			
			Tensor z = CGraph.op(CGraph.op(CGraph.op(x, 2, FunctionOPType.pow),three,SignOPType.multiplication),five,SignOPType.add);
			
			z.setGrad(new double[][][][] {{{{1}}}});
			
			z.backward();
			
			System.out.println(JsonUtils.toJson(x.getGrad()));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
}
