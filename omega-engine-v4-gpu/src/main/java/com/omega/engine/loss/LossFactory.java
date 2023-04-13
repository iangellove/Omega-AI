package com.omega.engine.loss;

import com.omega.yolo.loss.YoloLoss;
import com.omega.yolo.loss.YoloLoss2;
import com.omega.yolo.loss.YoloLoss3;

/**
 * LossFactory
 * @author Administrator
 *
 */
public class LossFactory {
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static LossFunction create(LossType type) {
		//square_loss,cross_entropy,softmax_with_cross_entropy
		switch (type) {
		case square_loss:
			return new SquareLoss();
		case cross_entropy:
			return new CrossEntropyLoss();
		case softmax_with_cross_entropy:
			return new CrossEntropyLoss2();
		case yolo:
			return new YoloLoss();
		case yolo2:
			return new YoloLoss2();
		case yolo3:
			return new YoloLoss3();
		default:
			return null;
		}
		
	}
	
}
