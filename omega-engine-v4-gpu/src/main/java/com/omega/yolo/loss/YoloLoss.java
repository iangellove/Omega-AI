package com.omega.yolo.loss;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.loss.LossFunction;
import com.omega.engine.loss.LossType;

/**
 * YoloLoss
 * 
 * @author Administrator
 *
 */
public class YoloLoss extends LossFunction {
	
	public final LossType lossType = LossType.yolo;
	
	private static YoloLoss instance;
	
	private int grid_number = 7;
	
	private int class_number = 20;
	
	private int once_size = grid_number * grid_number * 30;
	
	private Tensor loss;
	
	private Tensor diff;
	
	public static YoloLoss operation() {
		if(instance == null) {
			instance = new YoloLoss();
		}
		return instance;
	}
	
	public void init(Tensor input) {
		if(loss == null || diff.number != input.number) {
			this.loss = new Tensor(1, 1, 1, 1);
			this.diff = new Tensor(input.number, input.channel, input.height, input.width, true);
		}
	}
	
	/**
	 * loss = coor_error + iou_error + class_error
	 */
	@Override
	public Tensor loss(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		
		init(x);
		
		if(x.isHasGPU()) {
			x.syncHost();
		}
		
		float coor_loss = 0.0f;
		
		this.diff.data = new float[this.diff.data.length];
		
		for(int b = 0;b<x.getNumber();b++) {
			for(int gridIdx = 0;gridIdx<grid_number*grid_number;gridIdx++) {
				
				int bg_idx = b * once_size + gridIdx * 30;
				
				/**
				 * label has obj
				 */
				if(label.data[bg_idx + 0] == 1.0f) {
					
					int bestIndex = 0;
					
					float[] ious = iou(b, gridIdx, x.data, label.data);
					
					if(ious[0] > 0 || ious[1] > 0) {
						
						if(ious[1] > ious[0]) {
							bestIndex = 1;
						}
						
					}else {

						float[] rmse = box_rmse(b, gridIdx, x.data, label.data);
						
						if(rmse[1] < rmse[0]) {
							bestIndex = 1;
						}
						
					}
					
					if(bestIndex == 0) {
						coor_loss += 5.0f * Math.pow((1.0f - x.data[bg_idx + 0]), 2.0f);
						coor_loss += Math.pow((1.0f - ious[0]), 2.0f);

						this.diff.data[bg_idx + 0] = 5.0f * (1.0f - x.data[bg_idx + 0]);
						this.diff.data[bg_idx + 1] = 5.0f * (label.data[bg_idx + 1] - x.data[bg_idx + 1]);
						this.diff.data[bg_idx + 2] = 5.0f * (label.data[bg_idx + 2] - x.data[bg_idx + 2]);
						this.diff.data[bg_idx + 3] = (float) (5.0f * (Math.sqrt(label.data[bg_idx + 3]) - x.data[bg_idx + 3]));
						this.diff.data[bg_idx + 4] = (float) (5.0f * (Math.sqrt(label.data[bg_idx + 4]) - x.data[bg_idx + 4]));
					}else {
						coor_loss += 5.0f * Math.pow((1.0f - x.data[bg_idx + 5]), 2.0f);
						coor_loss += Math.pow((1.0f - ious[1]), 2.0f);

						this.diff.data[bg_idx + 5] = 5.0f * (1.0f - x.data[bg_idx + 5]);
						this.diff.data[bg_idx + 6] = 5.0f * (label.data[bg_idx + 6] - x.data[bg_idx + 6]);
						this.diff.data[bg_idx + 7] = 5.0f * (label.data[bg_idx + 7] - x.data[bg_idx + 7]);
						this.diff.data[bg_idx + 8] = (float) (5.0f * (Math.sqrt(label.data[bg_idx + 8]) - x.data[bg_idx + 8]));
						this.diff.data[bg_idx + 9] = (float) (5.0f * (Math.sqrt(label.data[bg_idx + 9]) - x.data[bg_idx + 9]));
					}
					
					/**
					 * class loss
					 */
					for(int cn = 0;cn<class_number;cn++) {
						int idx = bg_idx + 10 + cn;
						coor_loss += (label.data[idx] - x.data[idx]) * (label.data[idx] - x.data[idx]);
						this.diff.data[idx] = (label.data[idx] - x.data[idx]);
					}
				}else {
					/**
					 * not has obj
					 */
					coor_loss += (float) (0.5f * Math.pow(x.data[bg_idx + 0], 2.0) + 0.5f * Math.pow(x.data[bg_idx + 5], 2.0f));
					this.diff.data[bg_idx + 0] = 0.5f * (0.0f - x.data[bg_idx + 0]);
					this.diff.data[bg_idx + 5] = 0.5f * (0.0f - x.data[bg_idx + 5]);
				}
				
			}
			
		}
//		System.out.println(coor_loss + obj_confi_loss + noobj_confi_loss + class_loss);
		this.loss.data[0] = coor_loss;
//		System.out.println("out yolo loss.");
//		if(this.loss.data[0] > 1000) {
			System.out.println(coor_loss);
//			System.out.println(JsonUtils.toJson(x.data));
//			System.out.println(JsonUtils.toJson(diff.data));
//		}
		return loss;
	}

	@Override
	public Tensor diff(Tensor x, Tensor label) {
		// TODO Auto-generated method stub
		if(diff.isHasGPU()) {
			diff.hostToDevice();
		}
		return diff;
	}

	@Override
	public LossType getLossType() {
		// TODO Auto-generated method stub
		return LossType.yolo;
	}
	
	public float iou(float[] bbox1,float[] bbox2) {
		
		float iou = 0.0f;
		
		float[] intersect_bbox = new float[4];
		
		intersect_bbox[0] = Math.max(bbox1[0], bbox2[0]);
		intersect_bbox[1] = Math.max(bbox1[1], bbox2[1]);
		intersect_bbox[2] = Math.max(bbox1[2], bbox2[2]);
		intersect_bbox[3] = Math.max(bbox1[3], bbox2[3]);
		
		float w = Math.max(intersect_bbox[2] - intersect_bbox[0], 0);
		float h = Math.max(intersect_bbox[3] - intersect_bbox[1], 0);
		
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float area_intersect = w * h;
		iou = area_intersect / (area1 + area2 - area_intersect);
		
		return iou;
	}
	
	public float[] iou(int b,int gridIdx,float[] predBBox,float[] labelBBox) {
		
		float iou[] = new float[2];
		
		float[] b1 = new float[] {
				predBBox[b * once_size + gridIdx * 30 + 1] / grid_number - predBBox[b * once_size + gridIdx * 30 + 3] * predBBox[b * once_size + gridIdx * 30 + 3] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 2] / grid_number - predBBox[b * once_size + gridIdx * 30 + 4] * predBBox[b * once_size + gridIdx * 30 + 4] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 1] / grid_number + predBBox[b * once_size + gridIdx * 30 + 3] * predBBox[b * once_size + gridIdx * 30 + 3] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 2] / grid_number + predBBox[b * once_size + gridIdx * 30 + 4] * predBBox[b * once_size + gridIdx * 30 + 4] / 2.0f
		};
		
		float[] b2 = new float[] {
				predBBox[b * once_size + gridIdx * 30 + 6] / grid_number - predBBox[b * once_size + gridIdx * 30 + 8] * predBBox[b * once_size + gridIdx * 30 + 8] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 7] / grid_number - predBBox[b * once_size + gridIdx * 30 + 9] * predBBox[b * once_size + gridIdx * 30 + 9] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 6] / grid_number + predBBox[b * once_size + gridIdx * 30 + 8] * predBBox[b * once_size + gridIdx * 30 + 8] / 2.0f,
				predBBox[b * once_size + gridIdx * 30 + 7] / grid_number + predBBox[b * once_size + gridIdx * 30 + 9] * predBBox[b * once_size + gridIdx * 30 + 9] / 2.0f
		};
		
		float[] bl = new float[] {
				labelBBox[b * once_size + gridIdx * 30 + 1] / grid_number - labelBBox[b * once_size + gridIdx * 30 + 3] / 2.0f,
				labelBBox[b * once_size + gridIdx * 30 + 2] / grid_number - labelBBox[b * once_size + gridIdx * 30 + 4] / 2.0f,
				labelBBox[b * once_size + gridIdx * 30 + 1] / grid_number + labelBBox[b * once_size + gridIdx * 30 + 3] / 2.0f,
				labelBBox[b * once_size + gridIdx * 30 + 2] / grid_number + labelBBox[b * once_size + gridIdx * 30 + 4] / 2.0f
		};
		
		
//		System.out.println(JsonUtils.toJson(b1));
//		System.out.println(JsonUtils.toJson(b2));
//		System.out.println(JsonUtils.toJson(bl));
		
		iou[0] = iou(b1, bl);
		iou[1] = iou(b2, bl);
		return iou;
	}
	
	public float[] box_rmse(int b,int gridIdx,float[] predBBox,float[] labelBBox){
		
		float rmse[] = new float[2];
		
		float ax1 = predBBox[b * once_size + gridIdx * 30 + 1] / grid_number;
		float ay1 = predBBox[b * once_size + gridIdx * 30 + 2] / grid_number;
		float aw1 = predBBox[b * once_size + gridIdx * 30 + 3] * predBBox[b * once_size + gridIdx * 30 + 3];
		float ah1 = predBBox[b * once_size + gridIdx * 30 + 4] * predBBox[b * once_size + gridIdx * 30 + 4];
		
		float ax2 = predBBox[b * once_size + gridIdx * 30 + 6] / grid_number;
		float ay2 = predBBox[b * once_size + gridIdx * 30 + 7] / grid_number;
		float aw2 = predBBox[b * once_size + gridIdx * 30 + 8] * predBBox[b * once_size + gridIdx * 30 + 8];
		float ah2 = predBBox[b * once_size + gridIdx * 30 + 9] * predBBox[b * once_size + gridIdx * 30 + 9];
		
		float bx = labelBBox[b * once_size + gridIdx * 30 + 1] / grid_number;
		float by = labelBBox[b * once_size + gridIdx * 30 + 2] / grid_number;
		float bw = labelBBox[b * once_size + gridIdx * 30 + 3];
		float bh = labelBBox[b * once_size + gridIdx * 30 + 4];
		
		rmse[0] = (float) Math.sqrt((ax1 - bx) * (ax1 - bx) + (ay1 - by) * (ay1 - by) + (aw1 - bw) * (aw1 - bw) + (ah1 - bh) * (ah1 - bh));
		rmse[1] = (float) Math.sqrt((ax2 - bx) * (ax2 - bx) + (ay2 - by) * (ay2 - by) + (aw2 - bw) * (aw2 - bw) + (ah2 - bh) * (ah2 - bh));
	    return rmse;
	}
	
}
