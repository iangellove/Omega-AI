package com.omega.engine.loss;

import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.network.Network;
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
		case MSE:
			return new MSELoss();
		case BCE:
			return new BCELoss();
		case cross_entropy:
			return new CrossEntropyLoss();
		case softmax_with_cross_entropy:
			return new CrossEntropyLoss2();
		case yolo:
			return new YoloLoss(1);
		case yolov3:
			return null;
		case multiLabel_soft_margin:
			return new MultiLabelSoftMargin();
		default:
			return null;
		}
		
	}
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static LossFunction create(LossType type,int classNum) {
		//square_loss,cross_entropy,softmax_with_cross_entropy
		switch (type) {
		case MSE:
			return new MSELoss();
		case BCE:
			return new BCELoss();
		case cross_entropy:
			return new CrossEntropyLoss();
		case softmax_with_cross_entropy:
			return new CrossEntropyLoss2();
		case yolo:
			return new YoloLoss(classNum);
		case yolov3:
			return null;
		case multiLabel_soft_margin:
			return new MultiLabelSoftMargin();
		default:
			return null;
		}
		
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs,Network net) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case MSE:
				losses[i] = new MSELoss();
				break;
			case BCE:
				losses[i] = new BCELoss();
			case cross_entropy:
				losses[i] = new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] = new CrossEntropyLoss2();
				break;
			case yolo:
				losses[i] = new YoloLoss(1);
				break;
			case yolov2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.height, yolo2.network.width, yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolov3:
				YoloLayer yolo = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss3(yolo.class_number, yolo.bbox_num, yolo.mask, yolo.anchors, yolo.network.height, yolo.network.width, yolo.maxBox, yolo.total, yolo.ignoreThresh, yolo.truthThresh);
				break;
			case multiLabel_soft_margin:
				losses[i] = new MultiLabelSoftMargin();
				break;
			default:
				break;
			}
			if(losses[i] != null) {
				losses[i].net = net;
			}
		}
		
		return losses;
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs,int classNum,Network net) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case MSE:
				losses[i] = new MSELoss();
				break;
			case BCE:
				losses[i] = new BCELoss();
				break;
			case cross_entropy:
				losses[i] = new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] = new CrossEntropyLoss2();
				break;
			case yolo:
				losses[i] = new YoloLoss(classNum);
				break;
			case yolov2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.height, yolo2.network.width, yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolov3:
				YoloLayer yolo = (YoloLayer) outputs.get(i);
				losses[i] = new YoloLoss3(yolo.class_number, yolo.bbox_num, yolo.mask, yolo.anchors, yolo.network.height, yolo.network.width, yolo.maxBox, yolo.total, yolo.ignoreThresh, yolo.truthThresh);
				break;
			case multiLabel_soft_margin:
				losses[i] = new MultiLabelSoftMargin();
				break;
			default:
				break;
			}
			if(losses[i] != null) {
				losses[i].net = net;
			}
		}
		
		return losses;
	}
	
}
