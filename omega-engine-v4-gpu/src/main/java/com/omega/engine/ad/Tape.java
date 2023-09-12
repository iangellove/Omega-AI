package com.omega.engine.ad;

import java.io.Serializable;

import com.omega.common.data.Tensor;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;
import com.omega.engine.ad.op.data.GetOP;

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

	private Tensor x;
	
	private Tensor y;
	
	private Tensor output;
	
	private int[] position;
	
	private OP op;
	
	private float scalar;
	
	private Tensor tmp;
	
	private boolean sub = false;
	
	public Tape(OP op,Tensor self,Tensor other,float scalar,int[] position,Graph g) {
		this.setX(self);
		this.setY(other);
		if(position!=null) {
			int dims = position[0];
			if(!op.getOpType().equals(OPType.sum)) {
				int count = position[2];
				switch (dims) {
				case 0:
					setOutput(new Tensor(count, self.channel, self.height, self.width, self.isHasGPU(), g));
					break;
				case 1:
					setOutput(new Tensor(self.number, count, self.height, self.width, self.isHasGPU(), g));
					break;
				}
			}else{
				switch (dims) {
				case 0:
					setOutput(new Tensor(1, 1, 1, 1, self.isHasGPU(), g));
					break;
				case 1:
					setOutput(new Tensor(self.number, 1, 1, 1, self.isHasGPU(), g));
					break;
				}
			}
			
		}else {
			setOutput(new Tensor(self.number, self.channel, self.height, self.width, self.isHasGPU(), g));
		}
		this.setOp(op);
		this.scalar = scalar;
		this.setPosition(position);
	}

	public OP getOp() {
		return op;
	}

	public void setOp(OP op) {
		this.op = op;
	}
	
	public void zeroGrad() {
		if(getX().isRequiresGrad()) {
			getX().zeroGrad();
		}
		if(getY() != null && getY().isRequiresGrad()) {
			getY().zeroGrad();
		}
		if(getOutput().isRequiresGrad()) {
			getOutput().zeroGrad();
		}
	}
	
	public Tensor forward() {
		return this.op.forward(this);
	}
	
	public void backward(Tensor delta) {
		op.backward(delta, this);
//		if(this.getPosition() != null) {
//			GetOP getOp = (GetOP) op;
//			getOp.backward(delta, this);
//		}else {
//			op.backward(delta, this);
//		}
	}
	
	public void backward() {
		this.backward(getOutput().getGrad());
	}

	public float getScalar() {
		return scalar;
	}

	public void setScalar(float scalar) {
		this.scalar = scalar;
	}

	public int[] getPosition() {
		return position;
	}

	public Tensor getX() {
		return x;
	}

	public void setX(Tensor x) {
		this.x = x;
	}

	public Tensor getY() {
		return y;
	}

	public void setY(Tensor y) {
		this.y = y;
	}

	public Tensor getOutput() {
		return output;
	}

	public void setOutput(Tensor output) {
		this.output = output;
	}

	public void setPosition(int[] position) {
		this.position = position;
	}

	public Tensor getTmp() {
		if(tmp == null) {
			this.tmp = new Tensor(this.x.number, this.x.channel, this.x.height, this.x.width, this.x.isHasGPU());
		}
		return tmp;
	}

	public void setTmp(Tensor tmp) {
		this.tmp = tmp;
	}

	public boolean isSub() {
		return sub;
	}

	public void setSub(boolean sub) {
		this.sub = sub;
	}
	
}
