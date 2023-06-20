package com.omega.engine.loss;

import java.util.List;

import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.YoloLayer;
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
			return new YoloLoss(1);
		case yolo3:

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
		case square_loss:
			return new SquareLoss();
		case cross_entropy:
			return new CrossEntropyLoss();
		case softmax_with_cross_entropy:
			return new CrossEntropyLoss2();
		case yolo:
			return new YoloLoss(classNum);
		case yolo3:

		default:
			return null;
		}
		
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case square_loss:
				losses[i] = new SquareLoss();
				break;
			case cross_entropy:
				losses[i] =  new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] =  new CrossEntropyLoss2();
				break;
			case yolo:
				losses[i] =  new YoloLoss(1);
				break;
			case yolo2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] =  new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.height, yolo2.network.width, yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolo3:
				YoloLayer yolo = (YoloLayer) outputs.get(i);
				losses[i] =  new YoloLoss3(yolo.class_number, yolo.bbox_num, yolo.mask, yolo.anchors, yolo.network.height, yolo.network.width, yolo.maxBox, yolo.total, yolo.ignoreThresh, yolo.truthThresh);
				break;
			default:
				break;
			}
		}
		
		return losses;
	}
	
	public static LossFunction[] create(LossType type,List<Layer> outputs,int classNum) {
		LossFunction[] losses = new LossFunction[outputs.size()];
		
		for(int i = 0;i<outputs.size();i++) {

			switch (type) {
			case square_loss:
				losses[i] = new SquareLoss();
				break;
			case cross_entropy:
				losses[i] =  new CrossEntropyLoss();
				break;
			case softmax_with_cross_entropy:
				losses[i] =  new CrossEntropyLoss2();
				break;
			case yolo:
				losses[i] =  new YoloLoss(classNum);
				break;
			case yolo2:
				YoloLayer yolo2 = (YoloLayer) outputs.get(i);
				losses[i] =  new YoloLoss2(yolo2.class_number, yolo2.bbox_num, yolo2.anchors, yolo2.network.height, yolo2.network.width, yolo2.maxBox, yolo2.total, yolo2.ignoreThresh, yolo2.truthThresh);
				break;
			case yolo3:
				YoloLayer yolo = (YoloLayer) outputs.get(i);
				losses[i] =  new YoloLoss3(yolo.class_number, yolo.bbox_num, yolo.mask, yolo.anchors, yolo.network.height, yolo.network.width, yolo.maxBox, yolo.total, yolo.ignoreThresh, yolo.truthThresh);
				break;
			default:
				break;
			}
		}
		
		return losses;
	}
	
}
