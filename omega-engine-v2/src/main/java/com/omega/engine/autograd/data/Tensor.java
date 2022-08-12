package com.omega.engine.autograd.data;

import java.io.Serializable;
import com.omega.engine.autograd.exceptions.AutogradException;
import com.omega.engine.autograd.operater.OP;

public class Tensor implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -217717238613935294L;
	
	private String name;
	
	private double[][][][] data;
	
	private boolean requiresGrad = true;
	
	private double[][][][] grad;
	
	private OP op;
	
	private Tensor leftTensor;
	
	private Tensor rightTensor;
	
	private double e;
	
	public Tensor(double[][][][] data) {
		this.data = data;
		this.grad = new double[data.length][data[0].length][data[0][0].length][data[0][0][0].length];
		//this.requiresGrad = false;
	}
	
	public Tensor(double[][][][] data,boolean requiresGrad) {
		this.data= data;
		this.grad = new double[data.length][data[0].length][data[0][0].length][data[0][0][0].length];
		this.requiresGrad = requiresGrad;
	}
	
	public Tensor(double[][][][] data,Tensor left,OP op,boolean requiresGrad) throws AutogradException {
		this.data= data;
		this.grad = new double[data.length][data[0].length][data[0][0].length][data[0][0][0].length];
		this.setLeftTensor(left);
		this.setOp(op);
		this.requiresGrad = requiresGrad;
	}
	
	public Tensor(double[][][][] data,Tensor left,double e,OP op,boolean requiresGrad) throws AutogradException {
		this.data= data;
		this.grad = new double[data.length][data[0].length][data[0][0].length][data[0][0][0].length];
		this.setLeftTensor(left);
		this.e = e;
		this.setOp(op);
		this.requiresGrad = requiresGrad;
	}
	
	public Tensor(double[][][][] data,Tensor left,Tensor right,OP op,boolean requiresGrad) throws AutogradException {
		this.data = data;
		this.grad = new double[data.length][data[0].length][data[0][0].length][data[0][0][0].length];
		this.setLeftTensor(left);
		this.setRightTensor(right);
		this.setOp(op);
		this.requiresGrad = requiresGrad;
	}
	
	public double[][][][] getData() {
		return data;
	}

	public void setData(double[][][][] data) {
		this.data = data;
	}

	public boolean isRequiresGrad() {
		return requiresGrad;
	}

	public void setRequiresGrad(boolean requiresGrad) {
		this.requiresGrad = requiresGrad;
	}

	public double[][][][] getGrad() {
		return grad;
	}

	public void setGrad(double[][][][] grad) {
		this.grad = grad;
	}

	public boolean isLeaf() {
		if(this.getLeftTensor() == null && this.getRightTensor() == null) {
			return false;
		}
		return true;
	}

	public void backward() {

		if(this.op != null) {
			this.op.backward(this);
		}
		
		if(this.getLeftTensor() != null) {
			this.getLeftTensor().backward();
		}
		
		if(this.getRightTensor() != null) {
			this.getRightTensor().backward();
		}
		
	}

	public OP getOp() {
		return op;
	}

	public void setOp(OP op) {
		this.op = op;
	}

	public Tensor getLeftTensor() {
		return leftTensor;
	}

	public void setLeftTensor(Tensor leftTensor) {
		this.leftTensor = leftTensor;
	}

	public Tensor getRightTensor() {
		return rightTensor;
	}

	public void setRightTensor(Tensor rightTensor) {
		this.rightTensor = rightTensor;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public double getE() {
		return e;
	}

	public void setE(double e) {
		this.e = e;
	}

}
