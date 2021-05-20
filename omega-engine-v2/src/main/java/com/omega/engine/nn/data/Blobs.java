package com.omega.engine.nn.data;

import com.omega.common.utils.MatrixOperation;

/**
 * Blob opt utils
 * @author Administrator
 *
 */
public class Blobs {
	
	/**
	 * create zero blob
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @return
	 */
	public static Blob blob(int number,int channel,int height,int width) {
		return new Blob(number, channel, height, width);
	}
	
	/**
	 * create zero blob
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @return
	 */
	public static Blob zero(int number,int channel,int height,int width,Blob blob) {
		if(blob == null || number != blob.number) {
			blob = Blobs.blob(number, channel, height, width);
		}else{
			blob.clear();
		}
		return blob;
	}
	
	/**
	 * create zero blob
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @return
	 */
	public static Blob zero(Blob blob) {
		return Blobs.blob(blob.number, blob.channel, blob.height, blob.width);
	}
	
	/**
	 * create blob for data
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @param data
	 * @return
	 */
	public static Blob blob(int number,int channel,int height,int width,double[][] data) {
		return new Blob(number, channel, height, width, data);
	}
	
	/**
	 * create blob for matrix
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @param matrix
	 * @return
	 */
	public static Blob blob(int number,int channel,int height,int width,double[][][][] matrix) {
		return new Blob(number, channel, height, width, matrix);
	}
	
	/**
	 * create blob for matrix
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @param matrix
	 * @return
	 */
	public static Blob blob(double[][][][] matrix) {
		return new Blob(matrix);
	}
	
	/**
	 * create blob for matrix
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @param matrix
	 * @return
	 */
	public static Blob blob(double[][][][] matrix,Blob blob) {
		if(blob == null) {
			return new Blob(matrix);
		}else {
			blob.maxtir = matrix;
			return blob;
		}
	}
	
	/**
	 * 
	 * @param number
	 * @param channel
	 * @param height
	 * @param width
	 * @param blob
	 * @return
	 */
	public static Blob transform(int number,int channel,int height,int width,Blob blob) {

		if(blob.number == number && blob.channel == channel && blob.height == height && blob.width == width) {
			return blob;
		}else {
			return Blobs.blob(number, channel, height, width, MatrixOperation.transform(blob.maxtir, number, channel, height, width));
		}
		
	}
	
	
}