package com.omega.yolo.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.RandomUtils;

public class AnchorBoxUtils {
	
	
	public static Tensor getAnchorBox(Tensor bboxs,int num_clusters) {
		Tensor boxs = translateBoxes(bboxs);
		
		int N = boxs.number;
		float[][] dis = new float[N][num_clusters];
		int[] last_clu = new int[N];
		
		Tensor clu = RandomUtils.random(boxs, num_clusters);

		while (true) {
			
			for(int n = 0;n<bboxs.number;n++) {
				dis[n] = MatrixOperation.subtraction(1.0f, iou(boxs.getByNumber(n), clu));
			}
			
			//获取类聚中心
			int[] nearest_clusters = argmin(dis);
			
			boolean isAll = true;
			for(int i = 0;i<nearest_clusters.length;i++) {
				if(last_clu[i] != nearest_clusters[i]) {
					isAll = false;
				}
			}
			
			if(isAll) {
				break;
			}
			
			for(int k = 0;k<num_clusters;k++) {
				clu.setByNumber(k, median(boxs, nearest_clusters, k));
			}
			
			/**
			 * copy nearest_clusters to last_clu
			 */
			System.arraycopy(nearest_clusters, 0, last_clu, 0, last_clu.length);
		}
		
		return clu;
	}
	
	public static float[] median(Tensor boxs,int[] nearest_clu,int cidx) {
		
		float[] result = new float[2];
		
		List<Float> w = new ArrayList<Float>();
		List<Float> h = new ArrayList<Float>();
		
		for(int n = 0;n<nearest_clu.length;n++) {
			
			if(nearest_clu[n] == cidx) {
				w.add(boxs.getByNumber(n)[0]);
				h.add(boxs.getByNumber(n)[1]);
			}
			
		}
		
		result[0] = median(w);
		result[1] = median(h);
		
		return result;
	}
	
	private static float median(List<Float> total) {

		float j = 0;

		//集合排序

		Collections.sort(total);

		int size = total.size();

		if(size % 2 == 1){

			j = total.get((size-1)/2);

		}else {

			j = ((total.get(size/2-1) + total.get(size/2)) / 2.0f);

		}

		return j;
	}

	
	public static int[] argmin(float[][] dis) {
		int[] arg = new int[dis.length];
		
		for(int n = 0;n<arg.length;n++) {
			arg[n] = minIndex(dis[n]);
		}
		return arg;
	}
	
	public static int minIndex(float[] x) {
		int index = 0;
		float temp = x[0];
		for(int i = 1;i<x.length;i++) {
			if(x[i] < temp) {
				temp = x[i];
				index = i;
			}
		}
		return index;
	}
	
	public static Tensor translateBoxes(Tensor boxs) {
		
		Tensor wh = new Tensor(boxs.number, 1, 1, boxs.width / 2);
		
		for(int n = 0;n<boxs.number;n++) {
			wh.data[n * wh.width + 0] = Math.abs(boxs.data[n * boxs.width + 2] - boxs.data[n * boxs.width + 0]);
			wh.data[n * wh.width + 1] = Math.abs(boxs.data[n * boxs.width + 3] - boxs.data[n * boxs.width + 1]);
		}
		
		return wh;
	}
	
	public static float[] iou(float[] box,Tensor clusters) {
		
		float[] iou = new float[clusters.number];
		
		for(int k = 0;k<clusters.number;k++) {
			float x = clusters.data[k * 2 + 0] <  box[0]? clusters.data[k * 2 + 0] : box[0];
			float y = clusters.data[k * 2 + 1] <  box[1]? clusters.data[k * 2 + 1] : box[1];
			float intersection = x * y;
			float box_area = box[0] * box[1];
			float cluster_area = clusters.data[k * 2 + 0] * clusters.data[k * 2 + 1];
			iou[k] = intersection / (box_area + cluster_area - intersection);
		}
	
		return iou;
	}
	
	public static void main(String[] args) {
		
		int k = 2;
		int n = 4;
		
		float[] data = new float[] {183, 63, 241, 112,26, 86, 79, 133,139, 108, 178, 148,20, 130, 63, 170};
		
		Tensor boxs = new Tensor(n, 1, 1, 4, data);
		
		Tensor anchors = getAnchorBox(boxs, k);
		
		for(int i = 0;i < k;i++) {
			System.out.println(JsonUtils.toJson(anchors.getByNumber(i)));
		}
		
		float[] data2 = new float[] {10,14,23,27,37,58,81,82,135,169,344,319};
		
		int[] data3 = new int[data2.length];
		
		for(int i = 0;i<data2.length;i++) {
			data3[i] = (int) (data2[i] / 416 * 256);
		}
		
		System.out.println(JsonUtils.toJson(data3));
		
	}
	
}
