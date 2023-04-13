package com.omega.yolo.utils;

public class YoloUtils {
	

	public static float box_rmse(float[] a,float[] b) {
		return (float) Math.sqrt(Math.pow(a[0]-b[0], 2) + 
				Math.pow(a[1]-b[1], 2) + 
				Math.pow(a[2]-b[2], 2) + 
				Math.pow(a[3]-b[3], 2));
	}
	
	public static float overlap(float x1,float w1,float x2,float w2) {
		float l1 = x1 - w1/2;   //边框1的左边坐标
		float l2 = x2 - w2/2;
		float left = l1 > l2 ? l1 : l2;    //重合部分左边（上边）的坐标
		float r1 = x1 + w1/2;
		float r2 = x2 + w2/2;
		float right = r1 > r2 ? r2 : r1;    //重合部分右边（下边）的坐标
	    return (right - left);
	}
	
	public static float box_intersection(float[] a, float[] b){
	    float w = overlap(a[0], a[2], b[0], b[2]);
	    float h = overlap(a[1], a[3], b[1], b[3]);
	    if(w < 0 || h < 0) return 0;
	    float area = w*h;
	    return area;
	}
	
	public static float box_union(float[] a,float[] b){
	    float i = box_intersection(a, b);
	    float u = a[2]*a[3] + b[2]*b[3] - i;
	    return u;
	}
	
	public static float box_iou(float[] a,float[] b) {
		
	    return box_intersection(a, b)/box_union(a, b);  //IOU=交集/并集
	}
	
	
}
