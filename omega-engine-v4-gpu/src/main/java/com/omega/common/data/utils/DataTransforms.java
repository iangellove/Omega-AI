package com.omega.common.data.utils;

import com.omega.common.data.Tensor;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.PrintUtils;
import com.omega.common.utils.RandomUtils;

/**
 * 数据预处理工具
 * @author Administrator
 *
 */
public class DataTransforms {
	
	public static void randomCrop(Tensor input,Tensor output,int ch,int cw,int padding) {
		
		int maxHeight = input.height + padding * 2 - ch;
		
		int maxWidth = input.width + padding * 2 - cw;
		
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
							output.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = 0.0f;
						}else {
							output.data[n * channel * ch * cw + c * ch * cw + h * cw + w] = input.data[n * channel * height * width + c * height * width + toh * width + tow];
						}
					}
				}
			}
			
		}
		
	}
	
	public static void randomHorizontalFilp(Tensor input,Tensor output) {
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int n = 0;n<input.number;n++) {
			
			if(Math.random() >= 0.5d) {
				for(int c = 0;c<input.channel;c++) {
					for(int h = 0;h<input.height;h++) {
						for(int w = 0;w<input.width / 2;w++) {
							int ow = input.width - 1 - w;
							output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + ow];
							output.data[n * channel * height * width + c * height * width + h * width + ow] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
						if(input.width % 2 == 1) {
							int w = width / 2;
							output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}else {
				for(int c = 0;c<input.channel;c++) {
					for(int h = 0;h<input.height;h++) {
						for(int w = 0;w<input.width;w++) {
							output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}
			
		}

	}
	
	public static void cutout(Tensor input,Tensor output,int len) {

		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int n = 0;n<input.number;n++) {
			
			int y = RandomUtils.getInstance().nextInt(height);
			int x = RandomUtils.getInstance().nextInt(width);
			
			int y1 = (int) (y - Math.floor(len / 2));
			int y2 = (int) (y + Math.floor(len / 2));
			int x1 = (int) (x - Math.floor(len / 2));
			int x2 = (int) (x + Math.floor(len / 2));

			for(int c = 0;c<input.channel;c++) {
				for(int h = 0;h<input.height;h++) {
					for(int w = 0;w<input.width;w++) {
						if(h >= y1 && h <= y2 && w >= x1 && w <= x2) {
							output.data[n * channel * height * width + c * height * width + h * width + w] = 0.0f;
						}else {
							output.data[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}
			
		}

	}
	
	public static void normalize(Tensor input,Tensor output,float[] mean,float[] std) {
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int i = 0;i<input.dataLength;i++) {
			int c = (i/height/width)%channel;
			output.data[i] = (input.data[i] - mean[c]) / std[c];
		}
	}
	
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
	
	public static void cutout(Tensor input,int len) {
		float[] out = new float[input.dataLength];
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int n = 0;n<input.number;n++) {
			
			int y = RandomUtils.getInstance().nextInt(height);
			int x = RandomUtils.getInstance().nextInt(width);
			
			int y1 = (int) (y - Math.floor(len / 2));
			int y2 = (int) (y + Math.floor(len / 2));
			int x1 = (int) (x - Math.floor(len / 2));
			int x2 = (int) (x + Math.floor(len / 2));
			
//			System.out.println(y1+","+y2+","+x1+","+x2);
			
			for(int c = 0;c<input.channel;c++) {
				for(int h = 0;h<input.height;h++) {
					for(int w = 0;w<input.width;w++) {
						if(h >= y1 && h <= y2 && w >= x1 && w <= x2) {
							out[n * channel * height * width + c * height * width + h * width + w] = 0.0f;
						}else {
							out[n * channel * height * width + c * height * width + h * width + w] = input.data[n * channel * height * width + c * height * width + h * width + w];
						}
					}
				}
			}
			
		}
		input.data = out;
	}
	
	public static void normalize(Tensor input,float[] mean,float[] std) {
		int channel = input.channel;
		int height = input.height;
		int width = input.width;
		for(int i = 0;i<input.dataLength;i++) {
			int c = (i/height/width)%channel;
			input.data[i] = (input.data[i] - mean[c]) / std[c];
		}
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
		
		Tensor output = new Tensor(1, 1,  9, 9);
		
//		DataTransforms.randomHorizontalFilp(input);
//		
//		float[][] out1 = MatrixUtils.transform(input.data, 9, 9);
//		
//		PrintUtils.printImage(out1);
		
//		DataTransforms.randomCrop(input, 9, 9, 2);
		
//		float[][] out2 = MatrixUtils.transform(input.data, 9, 9);
//		
//		PrintUtils.printImage(out2);
		
		DataTransforms.cutout(input, output, 4);
		
		float[][] out3 = MatrixUtils.transform(output.data, 9, 9);
		
		PrintUtils.printImage(out3);
		
		
	}
	
}