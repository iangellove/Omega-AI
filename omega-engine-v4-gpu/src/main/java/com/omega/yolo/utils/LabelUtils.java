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
	
	public static void loadBoxCSV(String labelPath,Tensor box) {
		
		try (FileInputStream fin = new FileInputStream(labelPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;
			int onceSize = box.channel * box.height * box.width;
	        while((strTmp = buffReader.readLine())!=null){
//	        	System.out.println(strTmp);
	        	if(idx > 0) {
		        	String[] list = strTmp.split(",");
		        	for(int i = 2;i<list.length;i++) {
		        		box.data[(idx-1) * onceSize + i-2] = Float.parseFloat(list[i]);
		        	}
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void loadLabelCSV(String labelPath,Tensor label,String[] idxs) {
		
		try (FileInputStream fin = new FileInputStream(labelPath);
			InputStreamReader reader = new InputStreamReader(fin);	
		    BufferedReader buffReader = new BufferedReader(reader);){
			
			String strTmp = "";
			int idx = 0;
			int onceSize = label.channel * label.height * label.width;
	        while((strTmp = buffReader.readLine())!=null){
//	        	System.out.println(strTmp);
	        	if(idx > 0) {
		        	String[] list = strTmp.split(",");
		        	idxs[idx-1] = list[0];
		        	for(int i = 1;i<list.length;i++) {
		        		label.data[(idx-1) * onceSize + i-1] = Float.parseFloat(list[i]);
		        	}
	        	}
	        	idx++;
	        }
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
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
		float py = cy / cellSize - gridy;
		
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
	public static float[] labelToYolo(int[][] bbox,int stride,int im_w) {
		
//		float[][] bbox = normalization(data);
		
		float[] target = new float[stride * stride * 25];

//		System.out.println(JsonUtils.toJson(bbox));
		
		for(int i = 0;i<bbox.length;i++) {
			
			float x1 = bbox[i][1];
			float y1 = bbox[i][2];
			float x2 = bbox[i][3];
			float y2 = bbox[i][4];
			
			float cx = (x1 + x2) / (2 * im_w);
			float cy = (y1 + y2) / (2 * im_w);
			
			float w = (x2 - x1) / im_w;
			float h = (y2 - y1) / im_w;

			int gridx = (int)(cx * stride);
			int gridy = (int)(cy * stride);
			
			float px = cx * stride - gridx;
			float py = cy * stride - gridy;
			
			int clazz = new Float(bbox[i][0]).intValue();
			
			/**
			 * c1
			 */
			target[gridx * stride * 25 + gridy * 25 + 0] = 1.0f;

			/**
			 * class
			 */
			target[gridx * stride * 25 + gridy * 25 + 1 + clazz] = 1.0f;

			/**
			 * x1,y1,w1,h1
			 */
			target[gridx * stride * 25 + gridy * 25 + 21 + 0] = px;
			target[gridx * stride * 25 + gridy * 25 + 21 + 1] = py;
			target[gridx * stride * 25 + gridy * 25 + 21 + 2] = w;
			target[gridx * stride * 25 + gridy * 25 + 21 + 3] = h;
			
		}
		
		return target;
	}
	
	public static float[] labelToYoloV3(int[][] bbox,int im_w) {
		
		float[] target = new float[5 * bbox.length];

		for(int i = 0;i<bbox.length;i++) {
			
			float clazz = new Float(bbox[i][0]).intValue();
			
			float cx = bbox[i][1];
			float cy = bbox[i][2];
			float w = bbox[i][3];
			float h = bbox[i][4];
			
//			cx = cx / im_w;
//			cy = cy / im_w;
//			w = w / im_w;
//			h = h / im_w;
			
			target[i * 5 + 0] = clazz;
			target[i * 5 + 1] = cx;
			target[i * 5 + 2] = cy;
			target[i * 5 + 3] = w;
			target[i * 5 + 4] = h;

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
