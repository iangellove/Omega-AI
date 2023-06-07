package com.omega.engine.ad;

import java.io.Serializable;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.ad.op.OP;

/**
 * 计算图节点
 * @author Administrator
 *
 */
public class Tape implements Serializable{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 9147342370353517536L;

	private List<Tensor> inputs;
	
	private List<Tensor> outputs;
	
	private OP op;
	
	public Tape(List<Tensor> inputs,List<Tensor> outputs,OP op) {
		this.setInputs(inputs);
		this.setOutputs(outputs);
		this.setOp(op);
	}

	public List<Tensor> getInputs() {
		return inputs;
	}

	public void setInputs(List<Tensor> inputs) {
		this.inputs = inputs;
	}

	public List<Tensor> getOutputs() {
		return outputs;
	}

	public void setOutputs(List<Tensor> outputs) {
		this.outputs = outputs;
	}

	public OP getOp() {
		return op;
	}

	public void setOp(OP op) {
		this.op = op;
	}
	
	public void zeroGrad() {
		for(Tensor input:inputs){
			if(input.isRequiresGrad()) {
				input.zeroGrad();
			}
		}
		for(Tensor output:outputs){
			if(output.isRequiresGrad()) {
				output.zeroGrad();
			}
		}
	}
	
	public void backward(float[] delta) {
		op.backward(delta, inputs);
	}
	
	public void backward() {
		if(outputs.get(0).getGrad() == null) {
			op.backward(MatrixUtils.one(inputs.get(0).dataLength), inputs);
		}else {
			op.backward(outputs.get(0).getGrad(), inputs);
		}
	}
	
}
