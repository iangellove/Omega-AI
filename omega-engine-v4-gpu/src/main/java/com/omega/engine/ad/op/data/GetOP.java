package com.omega.engine.ad.op.data;

import java.util.ArrayList;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.engine.ad.Graph;
import com.omega.engine.ad.Tape;
import com.omega.engine.ad.op.OP;
import com.omega.engine.ad.op.OPType;

/**
 * 获取指定向量数据
 * @author Administrator
 *
 */
public class GetOP extends OP{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7010180428917414516L;

	public static GetOP op = null;
	
	public static final OPType opt = OPType.get;
	
	public static GetOP getInstance() {
		if(op == null) {
			op = new GetOP();
			op.setOpType(opt);
		}
		return op;
	}
	
	public Tensor forward(Tensor self,int[] position) {
		Tensor y = getByPosition(self, position);
		if(self.isRequiresGrad()) {
			y.setRequiresGrad(true);
		}
		List<Tensor> inputs = new ArrayList<Tensor>(1);
		inputs.add(self);
		List<Tensor> outputs = new ArrayList<Tensor>(1);
		outputs.add(y);
		Tape tape = new Tape(inputs, outputs, this, position);
		Graph.add(tape);
		return y;
	}

	@Override
	public void backward(float[] delta, List<Tensor> inputs,float scalar) {
		// TODO Auto-generated method stub
		
	}
	
	public void backward(float[] delta, List<Tensor> inputs, int[] position) {
		// TODO Auto-generated method stub
		System.out.println("add-delta:"+JsonUtils.toJson(delta));
		if(inputs.get(0).isRequiresGrad()) {
			if(inputs.get(0).getGrad() != null) {
				int n = inputs.get(0).getNumber();
				int c = inputs.get(0).getChannel();
				int h = inputs.get(0).getHeight();
				int w = inputs.get(0).getWidth();
				MatrixOperation.add(inputs.get(0).getGrad(), delta, n, c, h, w, position);
			}else {
				inputs.get(0).setGrad(delta, position);
			}
		}
		System.out.println("add--d1:"+JsonUtils.toJson(inputs.get(0).getGrad()));
	}
	
	public static Tensor getByPosition(Tensor org,int[] position) {
		
		int dims = position[0];
		int start = position[1];
		int count = position[2];
		
		switch (dims) {
		case 0:
			return getByNumber(org, start, count);
		case 1:
			return getByChannel(org, start, count);
		default:
			return null;
		}
		
	}
	
	public static Tensor getByNumber(Tensor org,int start,int count) {
		
		assert org.getNumber() >= (start + count - 1);
		
		Tensor y = new Tensor(count, org.channel, org.height, org.width);
		
		System.arraycopy(org.data, start * org.channel * org.height * org.width, y.data, 0, y.dataLength);
		
		return y;
	}
	
	public static Tensor getByChannel(Tensor org,int start,int count) {
		
		assert org.getChannel() >= (start + count - 1);
		
		Tensor y = new Tensor(org.number, count, org.height, org.width);

		int size = org.height * org.width;
		for(int n = 0;n<org.number;n++) {
			int startIndex = n * org.channel * size + start * size;
			System.arraycopy(org.data, startIndex, y.data, n * count * size, count * size);
		}
		
		return y;
	}
	
}
