package com.omega.yolo.utils;

import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.yolo.model.YoloDetection;

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
	
	public static float overlapMin(float x1,float w1,float x2,float w2) {
		float l1 = x1 - w1/2;   //边框1的左边坐标
		float l2 = x2 - w2/2;
		float left = l1 < l2 ? l1 : l2;    //重合部分左边（上边）的坐标
		float r1 = x1 + w1/2;
		float r2 = x2 + w2/2;
		float right = r1 < r2 ? r2 : r1;    //重合部分右边（下边）的坐标
	    return (right - left);
	}
	
	public static float box_intersection(float[] a, float[] b){
	    float w = overlap(a[0], a[2], b[0], b[2]);
	    float h = overlap(a[1], a[3], b[1], b[3]);
	    if(w < 0 || h < 0) return 0;
	    float area = w*h;
	    return area;
	}
	
	public static float box_encolsed(float[] a, float[] b){
	    float w = overlapMin(a[0], a[2], b[0], b[2]);
	    float h = overlapMin(a[1], a[3], b[1], b[3]);
	    if(w < 0 || h < 0) return 0;
	    float area = w*h;
	    return area;
	}
	
	public static float box_union(float[] a,float[] b){
	    float i = box_intersection(a, b);
	    float u = a[2] * a[3] + b[2] * b[3] - i;
	    return u;
	}
	
	public static float box_iou(float[] a,float[] b) {
	    return box_intersection(a, b) / box_union(a, b);  //IOU=交集/并集
	}
	
	public static float box_giou(float[] a,float[] b) {
		//intersection area
		float interArea = box_intersection(a, b);
		float unionArea = box_union(a, b);
		float iou = interArea / unionArea;
		//enclosed area
		float encolseArea = box_encolsed(a, b);
		//compute giou
		return iou - 1.0f * (encolseArea - unionArea) / encolseArea;
	}
	
	public static int entryIndex(int batch,int w,int h,int location,int entry,int outputs,int class_number){
	    int n =   location / (w*h);
	    int loc = location % (w*h);
	    return batch*outputs + n*w*h*(4+class_number+1) + entry*w*h + loc;
	}
	
	public static YoloDetection[][] getYoloDetections(Tensor output,float[] anchors,int[] mask,int bbox_num,int outputs,int class_number,int orgW,int orgH,float thresh){
		
		YoloDetection[][] dets = new YoloDetection[output.number][output.height * output.width * bbox_num];
		
		if(output.isHasGPU()) {
			output.syncHost();
		}
//		System.out.println(JsonUtils.toJson(output.getByNumber(0)));
		
		int count = 0;

		for(int b = 0;b<output.number;b++) {
			
			for (int i = 0;i<output.height * output.width;i++){
		        int row = i / output.width;
		        int col = i % output.width;
		        int nCount = 0;
		        for(int n = 0;n<bbox_num;n++){
		        	
		        	int n_index = n*output.width*output.height + row*output.width + col;

		            int obj_index = entryIndex(b, output.width, output.height, n_index, 4, outputs, class_number);
		            float objectness = output.data[obj_index];
		            System.out.println(objectness);
		            if(objectness > thresh) {
		            	nCount++;
	//		            System.out.println(objectness);
			            int box_index = entryIndex(b, output.width, output.height, n_index, 0, outputs, class_number);
			            YoloDetection det = new YoloDetection(class_number);
			            det.setBbox(getYoloBox(output, anchors, mask[n], box_index, col, row, output.width, output.height, orgW, orgH, output.height * output.width));
			            det.setObjectness(objectness);
	//		            det.setClasses(classes);
			            for(int j = 0; j < class_number; ++j){
			                int class_index = entryIndex(b, output.width, output.height, n_index, 4 + 1 + j, outputs, class_number);
			                float prob = objectness*output.data[class_index];
			                det.getProb()[j] = (prob > thresh) ? prob : 0;
			            }
	//		            System.out.println(b+":"+det.getBbox()[0] + ":" + det.getBbox()[1] + ":" + det.getBbox()[2] + ":" + det.getBbox()[3]);
			            dets[b][i * bbox_num + n] = det;
		            }
		        }
		        if(nCount > 0) {
		        	count++;
		        }
		    }
			
		}
		
		System.out.println("testCount:"+count);
		
		return dets;
	}
	
	public static void correntYoloBoxes(List<YoloDetection> dets,int n,int w,int h, int orgw,int orgh,boolean relative) {
		
		int new_w = 0;
	    int new_h = 0;
	    
	    if (((float)orgw/w) < ((float)orgh/h)) {
	        new_w = orgw;
	        new_h = (h * orgw)/w;
	    } else {
	        new_h = orgh;
	        new_w = (w * orgh)/h;
	    }
	    
	    for (int i = 0; i < n; ++i){
	    	float[] bbox = dets.get(i).getBbox();
	    	bbox[0] = (float) ((bbox[0] - (orgw - new_w)/2./orgw) / ((float)new_w/orgw)); 
	    	bbox[1] = (float) ((bbox[1] - (orgh - new_h)/2./orgh) / ((float)new_h/orgh)); 
	    	bbox[2] *= (float)orgw/new_w;
	    	bbox[3] *= (float)orgh/new_h;
	        if(!relative){
	        	bbox[0] *= w;
	        	bbox[1] *= w;
	        	bbox[2] *= h;
	        	bbox[3] *= h;
	        }
	        dets.get(i).setBbox(bbox);
	    }
	    
	}
	
	/**
	 * 真实框w,h
	 * bh = ph * exp(th)
	 * bw = pw * exp(tw)
	 * ph,pw:锚框(anchor)
	 * th,tw:网络输出(锚框的比值)
	 * @param x
	 * @param anchors
	 * @param n
	 * @param index
	 * @param i
	 * @param j
	 * @param lw
	 * @param lh
	 * @param w
	 * @param h
	 * @param stride
	 * @return
	 */
	public static float[] getYoloBox(Tensor x,float[] anchors,int n,int index,int i,int j,int lw,int lh,int w,int h,int stride) {
		float[] box = new float[4];
		box[0] = (i + x.data[index + 0 * stride]) / lw;
		box[1] = (j + x.data[index + 1 * stride]) / lh;
		box[2] = (float) (Math.exp(x.data[index + 2 * stride]) * anchors[2 * n] / w);
		box[3] = (float) (Math.exp(x.data[index + 3 * stride]) * anchors[2 * n + 1] / h);
		return box;
	}
	
}
