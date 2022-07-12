package com.omega.common.utils;

import java.util.concurrent.RecursiveAction;

public class Dilation extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 359367116454030645L;

	private int start = 0;
	
	private int end = 0;
	
	private float[][][][] x;
	
	private float[][][][] y;
	
	private int stride = 2;
	
	private int N = 0;
	private int C = 0;
	private int H = 0;
	private int W = 0;
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		
	}
	
	public void dilation() {
		
		int s = stride - 1;
		
		for(int n = 0;n<N;n++) {
			
			for(int c = 0;c<C;c++) {
				
				for(int h = 0;h<H;h++) {
					
					for(int w = 0;w<W;w++) {
						
//						if() {
//							
//						}
						
						y[n][c][h + s][w + s] = y[n][c][h][w];
						
					}
					
				}
				
			}
			
		}
		
		
		
	}
	
}
