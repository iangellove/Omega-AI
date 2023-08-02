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
	yolo2,
	yolo3,
	MSE,
	BCE
	
}
