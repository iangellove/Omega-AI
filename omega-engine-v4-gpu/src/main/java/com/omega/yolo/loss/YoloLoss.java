package com.omega.yolo.loss;

import com.omega.common.data.Tensor;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;

/**
 * YoloLoss
 * 
 * @author Administrator
 *
 */
public class YoloLoss extends LossFunction {

	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.yolo;
	}
	
	
	
}
