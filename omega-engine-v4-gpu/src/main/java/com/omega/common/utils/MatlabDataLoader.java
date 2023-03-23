package com.omega.common.utils;

import java.io.IOException;
import java.util.Map;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLSingle;
import com.jmatio.types.MLUInt8;
import com.omega.engine.nn.data.DataSet;

public class MatlabDataLoader {
	
	public static DataSet loadMatData(String path) {
		
		try {
			
			MatFileReader read = new MatFileReader(path);
			
			Map<String, MLArray> map = read.getContent();

			MLArray images = map.get("images");
			
			MLArray targets = map.get("targets");
			
			int[] imagesDim = images.getDimensions();
			
			int[] targetsDim = targets.getDimensions();
			
			int number = imagesDim[0];
			
			int channel = imagesDim[1];
			
			int height = imagesDim[2];
			
			int width = imagesDim[3];
			
			int labelChannel = targetsDim[1];
			
			int labelSize = targetsDim[2];
			
			float[] data = new float[number * channel * height * width];
			
			MLUInt8 d = (MLUInt8)images;
			
			MLSingle t = (MLSingle)targets;
			
			for(int i = 0;i<number;i++) {
				for(int j = 0;j<channel * height * width;j++) {
					data[i * channel * height * width + j] = Integer.parseInt(d.get(i,j).toString()) / 255.0f;
					if(data[i * channel * height * width + j] <= 0) {
						data[i * channel * height * width + j] = 0.0f;
					}
				}
			}
			
			float[] label = new float[number * labelChannel * labelSize];
			
			for(int i = 0;i<number;i++) {
				for(int j = 0;j<labelSize;j++) {
					label[i * labelSize + j] = Float.parseFloat(t.get(i,j).toString());
				}
			}
			
			System.out.println(JsonUtils.toJson(imagesDim));
			
			System.out.println(JsonUtils.toJson(targetsDim));
			
			return new DataSet(number, channel, height, width, labelChannel, labelSize, data, label);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return null;
	}
	
}
