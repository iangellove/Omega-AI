package com.omega.common.data.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;

/**
 * 数据预处理工具
 * @author Administrator
 *
 */
public class DataTransforms {
	
	public static void randomCrop(Tensor input,int ch,int cw,int padding) {
		
		int maxHeight = input.height + padding * 2 - ch;
		
		int maxWidth = input.width + padding * 2 - cw;
		
		float[] out = new float[input.number * input.channel * ch * cw];
		
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int n = 0;n<input.number;n++) {
			
			int rh = (int) (Math.random() * (maxHeight + 1));
			
			int rw = (int) (Math.random() * (maxWidth + 1));
			
			for(int c = 0;c<channel;c++) {
				for(int h = 0;h<ch;h++) {
					for(int w = 0;w<cw;w++) {
						int oh = h + rh;
						int ow = w + rw;
						int toh = oh - padding;
						int tow = ow - padding;
						if(oh < padding || ow < padding || toh >= height || tow >= width) {
//							out[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
						}else {
							out[n * channel * ch * cw + c * ch * cw + h * cw + w] = input.data[n * channel * height * width + c * height * width + toh * width + tow];
						}
					}
				}
			}
			
		}
		input.data = out;
		
	}
	
	public static void randomHorizontalFilp(Tensor input) {
		float[] out = new float[input.dataLength];
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int n = 0;n<input.number;n++) {
			
			if(Math.random() >= 0.5d) {
				for(int c = 0;c<input.channel;c++) {
					for(int h = 0;h<input.height;h++) {
						for(int w = 0;w<input.width / 2;w++) {
							int ow = input.width - 1 - w;
							out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + ow];
							out[n * channel * height * width + c * height * width + h * width + ow] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
						if(input.width % 2 == 1) {
							int w = width / 2;
							out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}else {
				for(int c = 0;c<input.channel;c++) {
					for(int h = 0;h<input.height;h++) {
						for(int w = 0;w<input.width;w++) {
							out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}
			
		}
		input.data = out;
	}
	
	public static void main(String[] args) {
		
		float[][] x = new float[][] {
			{0,0,0,0,0,0,0,0,0},
			{0,0,1,1,1,1,0,0,0},
			{0,0,0,0,0,1,0,0,0},
			{0,0,0,0,1,0,0,0,0},
			{0,0,0,1,0,0,0,0,0},
			{0,0,0,1,0,0,0,0,0},
			{0,0,1,0,0,0,0,0,0},
			{0,1,1,1,1,1,1,0,0},
			{0,0,0,0,0,0,0,0,0}
			};
			
		float[][] x2 = new float[][] {
				{0,0,0,0,0,0,0,0},
				{0,0,1,1,1,1,0,0},
				{0,0,0,0,0,1,0,0},
				{0,0,0,0,1,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,1,0,0,0,0},
				{0,0,0,0,0,0,0,0}
			};
		
		float[] x1 = MatrixUtils.transform(x);
		
		Tensor input = new Tensor(1, 1,  9, 9, x1);
		
//		DataTransforms.randomHorizontalFilp(input);
//		
//		float[][] out1 = MatrixUtils.transform(input.data, 9, 9);
//		
//		PrintUtils.printImage(out1);
		
		DataTransforms.randomCrop(input, 9, 9, 2);
		
		float[][] out2 = MatrixUtils.transform(input.data, 9, 9);
		
		PrintUtils.printImage(out2);
		
	}
	
}
