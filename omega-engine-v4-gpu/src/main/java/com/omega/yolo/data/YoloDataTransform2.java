package com.omega.yolo.data;

import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.yolo.utils.YoloDecode;

/**
 * 
 * @author Administrator
 *
 */
public class YoloDataTransform2 extends DataTransform {
	
	private float jitter = 0.2f;
	
	private float hue = 0.1f;
	
	private float saturation = 0.75f;
	
	private float exposure = 0.75f;
	
	private int classnum = 1;
	
	private int numBoxes = 7;
	
	private DataType dataType;
	
	public YoloDataTransform2(int classnum,DataType dataType) {
		this.classnum = classnum;
		this.dataType = dataType;
	}
	
	@Override
	public void transform(Tensor input, Tensor label, String[] idxSet, int[] indexs, Map<String, float[]> orgLabelData) {
		// TODO Auto-generated method stub
		
		label.data = new float[label.dataLength];
		
		YoloDataTransformJob.transform(input, label, idxSet, indexs, orgLabelData, dataType);

	}

	@Override
	public void showTransform(String outputPath, Tensor input, Tensor label, String[] idxSet, int[] indexs,
			Map<String, float[]> orgLabelData) {
		// TODO Auto-generated method stub
//		Tensor oLabel = new Tensor(input.number, 1, 1, 5);
		
		this.transform(input, label, idxSet, indexs, orgLabelData);
		
		ImageUtils utils = new ImageUtils();
		
		input.data = MatrixOperation.multiplication(input.data, 255.0f);
		
//		oLabel.data = MatrixOperation.multiplication(oLabel.data, input.width);
		
		float[][][] draw_bbox = YoloDecode.getDetectionLabel(label, input.width, input.height);
		
		for(int b = 0;b<input.number;b++) {
			
			float[] once = input.getByNumber(b);
			
			float[] labelArray = null;
			
			for(float[] la:draw_bbox[b]) {
				if(la[0] == 1) {
					labelArray = la;
					break;
				}
			}
			
			int[][] bbox = null;
			
			if(labelArray != null) {
				bbox = new int[][] {
					{	
						0,
						(int) labelArray[2],
						(int) labelArray[3],
						(int) labelArray[4],
						(int) labelArray[5]
					}
			};
			}
			
//			System.out.println(JsonUtils.toJson(bbox));
			
			utils.createRGBImage(outputPath + b + ".png", "png", ImageUtils.color2rgb2(once, input.height, input.width), input.width, input.height, bbox);
			
		}
	}
	
}
