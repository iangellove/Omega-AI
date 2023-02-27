package com.omega.engine.loss;

import com.omega.common.data.Tensor;

/**
 * yolov1 loss
 * @author Administrator
 *
 */
public class DetectionLoss extends LossFunction {

	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		
		return null;
	}
	
	public Tensor loss_cpu(Tensor x, Tensor label) {
		
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
		return LossType.detection;
	}

}
