package com.omega.engine.loss;

/**
 * LossType
 * @author Administrator
 *
 */
public enum LossType {
	
	cross_entropy,
	softmax_with_cross_entropy,
	multiLabel_soft_margin,
	detection,
	yolo,
	yolov2,
	yolov3,
	yolov5,
	MSE,
	BCE,
	BCEWithLogits
	
}
