package com.omega.engine.model;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.YoloLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;

import jcuda.Sizeof;

/**
 * darknet weight file loader
 * @author Administrator
 *
 */
public class DarknetLoader {
	
	public static void loadWeight(Network net,String path,int layerCount) {
		
		System.out.println("start load weight.");
		
		try(RandomAccessFile file = new RandomAccessFile(path, "r")){	
			
			int major = readInt(file);
		    int minor = readInt(file);
		    int revision = readInt(file);

		    System.out.println("major:"+major);
		    System.out.println("minor:"+minor);
		    System.out.println("revision:"+revision);
		    
		    if((major * 10 + minor) >= 2){
		    	long seen = readBigInt(file);
		    	System.out.println("seen:"+seen);
		    }else {
		    	int seen = readInt(file);
		    	System.out.println("seen:"+seen);
		    }
		    
		    int index = 0;
		    
		    /**
		     * load weight by layers
		     */
		    for(int l = 0;l<net.layerList.size();l++) {
		    	
		    	Layer layer = net.layerList.get(l);
		    	
		    	if(index < layerCount) {

			    	switch (layer.getLayerType()) {
					case conv:

						loadConvWeights(file, l, layer, net.layerList);
						index++;
						break;
					case full:
						loadFullyWeights(file, l, layer, net.layerList);
						index++;
						break;
					case pooling:
						index++;
						break;
					default:
						break;
					}
			    	
		    	}
		    	
		    }

		    System.out.println("load weight finish.");
		    
		}catch (Exception e) {
			e.printStackTrace();
	    }
		
	}
	
	public static void loadWeight(Network net,String path) {
		
		System.out.println("start load weight.");
		
		try(RandomAccessFile file = new RandomAccessFile(path, "r")){	
			
			int major = readInt(file);
		    int minor = readInt(file);
		    int revision = readInt(file);

		    System.out.println("major:"+major);
		    System.out.println("minor:"+minor);
		    System.out.println("revision:"+revision);
		    
		    if((major * 10 + minor) >= 2){
		    	long seen = readBigInt(file);
		    	System.out.println("seen:"+seen);
		    }else {
		    	int seen = readInt(file);
		    	System.out.println("seen:"+seen);
		    }

		    /**
		     * load weight by layers
		     */
		    for(int l = 0;l<net.layerList.size();l++) {
		    	
		    	Layer layer = net.layerList.get(l);
		    	
		    	switch (layer.getLayerType()) {
				case conv:
					loadConvWeights(file, l, layer, net.layerList);
					break;
				case full:
					loadFullyWeights(file, l, layer, net.layerList);
					break;
				default:
					break;
				}
		    	
		    }

		    System.out.println("load weight finish.");
		    
		}catch (Exception e) {
			e.printStackTrace();
	    }
		
	}
	
	public static void loadConvWeights(RandomAccessFile inputStream,int index,Layer layer,List<Layer> layerList) throws IOException {
		
		if(layerList.get(index+1) instanceof YoloLayer) {
			
			int biasLength = 255;
			
			int weightLength = layer.weight.channel * 255 * layer.weight.height * layer.weight.width;

			/**
			 * load biases
			 */
			skipFloat(inputStream, biasLength);
			
			/**
			 * load conv weight
			 */
			skipFloat(inputStream, weightLength);
			
		}else {

			/**
			 * load biases
			 */
			readFloat(inputStream, layer.bias);
			/**
			 * load bn params
			 */
			if(!layer.hasBias && index < layerList.size() - 1 && layerList.get(index+1) instanceof BNLayer) {
				BNLayer bnl = (BNLayer) layerList.get(index+1);
				bnl.init();
				readFloat(inputStream, bnl.gamma);
				bnl.beta = layer.bias.copyGPU();
				readFloat(inputStream, bnl.runingMean);
				readFloat(inputStream, bnl.runingVar);
			}
			/**
			 * load conv weight
			 */
			readFloat(inputStream, layer.weight);
			
		}
		
	}
	
	public static void loadFullyWeights(RandomAccessFile inputStream,int index,Layer layer,List<Layer> layerList) throws IOException {
		
		/**
		 * load biases
		 */
		readFloat(inputStream, layer.bias);
		/**
		 * load conv weight
		 */
		readFloat(inputStream, layer.weight);
		/**
		 * load bn params
		 */
		if(!layer.hasBias && index < layerList.size() - 1) {
			BNLayer bnl = (BNLayer) layerList.get(index+1);
			bnl.init();
			readFloat(inputStream, bnl.gamma);
			readFloat(inputStream, bnl.runingMean);
			readFloat(inputStream, bnl.runingVar);
		}
		
	}
	
	public static long readBigInt(RandomAccessFile inputStream) throws IOException {
	    long retVal;
	    byte[] buffer = new byte[Sizeof.LONG];
	    inputStream.readFully(buffer);
	    ByteBuffer wrapped = ByteBuffer.wrap(buffer);
	    wrapped.order(ByteOrder.LITTLE_ENDIAN);
	    retVal = wrapped.getLong();
	    return retVal;
	}
	
	public static int readInt(RandomAccessFile inputStream) throws IOException {
	    int retVal;
	    byte[] buffer = new byte[Sizeof.INT];
	    inputStream.readFully(buffer);
	    ByteBuffer wrapped = ByteBuffer.wrap(buffer);
	    wrapped.order(ByteOrder.LITTLE_ENDIAN);
	    retVal = wrapped.getInt();
	    return retVal;
	}
	
	public static float readFloat(RandomAccessFile inputStream) throws IOException {
		float retVal;
	    byte[] buffer = new byte[Sizeof.FLOAT];
	    inputStream.readFully(buffer);
	    ByteBuffer wrapped = ByteBuffer.wrap(buffer);
	    wrapped.order(ByteOrder.LITTLE_ENDIAN);
	    retVal = wrapped.getFloat();
	    return retVal;
	}
	
	public static void skipFloat(RandomAccessFile inputStream,int length) throws IOException {
		byte[] buffer = new byte[length * Sizeof.FLOAT];
	    inputStream.readFully(buffer);
	    ByteBuffer wrapped = ByteBuffer.wrap(buffer);
	    wrapped.order(ByteOrder.LITTLE_ENDIAN);
	    wrapped.getFloat();
	}
	
	public static void readFloat(RandomAccessFile inputStream,Tensor data) throws IOException {
		for(int i = 0;i<data.data.length;i++) {
			data.data[i] = readFloat(inputStream);
		}
		if(data.isHasGPU()) {
			data.hostToDevice();
		}
	}
	
	public static void readFloat(RandomAccessFile inputStream,float[] data) throws IOException {
		for(int i = 0;i<data.length;i++) {
			data[i] = readFloat(inputStream);
		}
	}
	
}
