package com.omega.yolo.utils;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.math.BigDecimal;

import com.omega.common.data.Tensor;

/**
 * yolo label transform to the location
 * @author Administrator
 *
 */
public class LabelUtils {
	
	public static void loadLabel(String labelPath,Tensor label) {
		
		try (FileInputStream fin = new FileInputStream(labelPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;
			int onceSize = label.channel * label.height * label.width;
	        while((strTmp = buffReader.readLine())!=null){
	        	String[] list = strTmp.split(" ");
	        	for(int i = 1;i<list.length;i++) {
	        		label.data[idx * onceSize + i-1] = Float.parseFloat(list[i]);
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	/**
	 * labelToLocation
	 * @param wmax
	 * @param wmin
	 * @param hmax
	 * @param hmin
	 * @param cla = 20
	 * @param stride = 7
	 * @param wimg = 448
	 * @param himg = 448
	 * @return 7 * 7 * (2 + 8 + 20) = 1470
	 * target = [cx1,cy1,w1,h1,c1,cx2,cy2,w2,h2,c2,clazz1.....,clazz20]
	 * w = wmax - wmin
	 * h = hmax - hmin
	 * cx = (wmax + wmin) / 2
	 * cy = (hmax + hmin) / 2
	 * gridx = int(cx / stride)
	 * gridy = int(cy / stride)
	 * x = (cx - (gridx * cellSize)) / cellSize
	 * y = (cy - (gridy * cellSize)) / cellSize
	 */
	public static float[] labelToLocation(int wmax,int wmin,int hmax,int hmin,int cla,int stride) {
		
		float cellSize = 1.0f / stride;
		
		float[] target = new float[stride * stride * 30];
		
		float w = wmax - wmin;
		float h = hmax - hmin;
		
		float cx = (wmax + wmin) / 2;
		float cy = (hmax + hmin) / 2;
		int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
		int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
		
		/**
		 * c1
		 */
		target[gridx * stride * 30 + gridy * 30 + 0] = 1.0f;
		/**
		 * c2
		 */
		target[gridx * stride * 30 + gridy * 30 + 5] = 1.0f;
		
		float x = cx / cellSize - gridx;
		float y = cy / cellSize - gridx;
		
		/**
		 * x1,y1,w1,h1
		 */
		target[gridx * stride * 30 + gridy * 30 + 1] = x;
		target[gridx * stride * 30 + gridy * 30 + 2] = y;
		target[gridx * stride * 30 + gridy * 30 + 3] = w;
		target[gridx * stride * 30 + gridy * 30 + 4] = h;
		/**
		 * x2,y2,w2,h2
		 */
		target[gridx * stride * 30 + gridy * 30 + 6] = x;
		target[gridx * stride * 30 + gridy * 30 + 7] = y;
		target[gridx * stride * 30 + gridy * 30 + 8] = w;
		target[gridx * stride * 30 + gridy * 30 + 9] = h;
	
		/**
		 * class
		 */
		target[gridx * stride * 30 + gridy * 30 + cla + 9] = 1.0f;
		
		return target;
	} 
	
	/**
	 * labelToLocation
	 * @param cx
	 * @param cy
	 * @param w
	 * @param h
	 * @param cla = 20
	 * @param stride = 7
	 * @param wimg = 448
	 * @param himg = 448
	 * @return 7 * 7 * (2 + 8 + 20) = 1470
	 * target = [px1,py1,w1,h1,c1,px2,py2,w2,h2,c2,clazz1.....,clazz20]
	 * gridx = int(cx / stride)
	 * gridy = int(cy / stride)
	 * px = (cx - (gridx * cellSize)) / cellSize
	 * py = (cy - (gridy * cellSize)) / cellSize
	 */
	public static float[] labelToYolo(int cx,int cy,int w,int h,int cla,int stride) {
		
		float cellSize = 1.0f / stride;
		
		float[] target = new float[stride * stride * 30];
		
		int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
		int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue() - 1;
		
		/**
		 * c1
		 */
		target[gridx * stride * 30 + gridy * 30 + 0] = 1.0f;
		/**
		 * c2
		 */
		target[gridx * stride * 30 + gridy * 30 + 5] = 1.0f;
		
		float px = cx / cellSize - gridx;
		float py = cy / cellSize - gridx;
		
		/**
		 * x1,y1,w1,h1
		 */
		target[gridx * stride * 30 + gridy * 30 + 1] = px;
		target[gridx * stride * 30 + gridy * 30 + 2] = py;
		target[gridx * stride * 30 + gridy * 30 + 3] = w;
		target[gridx * stride * 30 + gridy * 30 + 4] = h;
		/**
		 * x2,y2,w2,h2
		 */
		target[gridx * stride * 30 + gridy * 30 + 6] = px;
		target[gridx * stride * 30 + gridy * 30 + 7] = py;
		target[gridx * stride * 30 + gridy * 30 + 8] = w;
		target[gridx * stride * 30 + gridy * 30 + 9] = h;
	
		/**
		 * class
		 */
		target[gridx * stride * 30 + gridy * 30 + cla + 9] = 1.0f;
		
		return target;
	} 
	
	/**
	 * labelToLocation
	 * @param cx
	 * @param cy
	 * @param w
	 * @param h
	 * @param cla = 20
	 * @param stride = 7
	 * @param wimg = 448
	 * @param himg = 448
	 * @return 7 * 7 * (2 + 8 + 20) = 1470
	 * target = [px1,py1,w1,h1,c1,px2,py2,w2,h2,c2,clazz1.....,clazz20]
	 * gridx = int(cx / stride)
	 * gridy = int(cy / stride)
	 * px = (cx - (gridx * cellSize)) / cellSize
	 * py = (cy - (gridy * cellSize)) / cellSize
	 */
	public static float[] labelToYolo(int[][] data,int stride) {
		
		float[][] bbox = normalization(data);
		
		float cellSize = 1.0f / stride;
		
		float[] target = new float[stride * stride * 30];

//		System.out.println(JsonUtils.toJson(bbox));
		
		for(int i = 0;i<bbox.length;i++) {
			
			float cx = bbox[i][1];
			float cy = bbox[i][2];
			float w = bbox[i][3];
			float h = bbox[i][4];
			int clazz = new Float(bbox[i][0]).intValue();
//			System.out.println(new BigDecimal(cx * 1.0f / 448.0f).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue());
			int gridx = new BigDecimal(cx).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue();
			int gridy = new BigDecimal(cy).divide(new BigDecimal(cellSize), BigDecimal.ROUND_CEILING).intValue();
			
//			System.out.println(cx+":"+cy+":"+gridx+":"+gridy);
			
			/**
			 * c1
			 */
			target[gridx * stride * 30 + gridy * 30 + 0] = 1.0f;
			/**
			 * c2
			 */
			target[gridx * stride * 30 + gridy * 30 + 5] = 1.0f;
			
			float px = (cx - gridx * cellSize) / cellSize;
			float py = (cy - gridy * cellSize) / cellSize;
			
			/**
			 * x1,y1,w1,h1
			 */
			target[gridx * stride * 30 + gridy * 30 + 1] = px;
			target[gridx * stride * 30 + gridy * 30 + 2] = py;
			target[gridx * stride * 30 + gridy * 30 + 3] = w;
			target[gridx * stride * 30 + gridy * 30 + 4] = h;
			/**
			 * x2,y2,w2,h2
			 */
			target[gridx * stride * 30 + gridy * 30 + 6] = px;
			target[gridx * stride * 30 + gridy * 30 + 7] = py;
			target[gridx * stride * 30 + gridy * 30 + 8] = w;
			target[gridx * stride * 30 + gridy * 30 + 9] = h;
		
			/**
			 * class
			 */
			target[gridx * stride * 30 + gridy * 30 + clazz + 10] = 1.0f;

		}
		
		return target;
	}
	
	public static float[][] normalization(int[][] data){
		
		float[][] bbox = new float[data.length][data[0].length];
		
		for(int i = 0;i<bbox.length;i++) {
			
			for(int j = 0;j<bbox[i].length;j++) {
				
				bbox[i][j] = data[i][j] * 1.0f / 448.0f;
				
			}
			
		}
		
		return bbox;
	}
	
}
