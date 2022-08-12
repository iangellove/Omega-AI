package com.omega.common.utils;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class Im2col extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private double[][][][] x;

	private double[][] colMatrix;

	private int fh, fw, start, end, stride, pad,oh,ow;

	public Im2col(double[][][][] x, int fh, int fw, int oh ,int ow,int stride, int pad, double[][] col, int start, int end) {
		// TODO Auto-generated constructor stub
		this.x = x;

		this.colMatrix = col;

		this.start = start;
		this.end = end;
		this.fh = fh;
		this.fw = fw;
		this.stride = stride;
		this.pad = pad;
		
		this.oh = oh;
		this.ow = ow;
	}

	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;

		if (length < 4 || length <= x.length / 4) {

			toCol();

		} else {

			int mid = (start + end + 1) >>> 1;
			Im2col left = new Im2col(x, fh, fw,oh,ow,stride, pad, colMatrix, start, mid - 1);
			Im2col right = new Im2col(x, fh, fw,oh,ow, stride, pad, colMatrix, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
		}
	}

	private void toCol() {
		
		for (int n = start; n <= end; n++) {
			
			int posX = 0;
			int posY = 0;
			
			for (int i = 0; i < oh; i++) {				
				for (int j = 0; j < ow; j++) {
					int pos = 0;					
					double[] col = colMatrix[n * oh * ow + i * oh + j];
					
					for (int c = 0; c < x[n].length; c++) {
						toCol(x[n][c], col, pos, posX, posY);
						pos += fh * fw;
					}
					posY = j + stride;
				}
				posX = i + stride;
				posY = 0;
			}
		}
	}

	private void toCol(double[][] s, double[] t, int pos, int i, int j) {
		for (int k = i; k < i + fh; k++) {
			System.arraycopy(s[k], j, t, pos, fw);
			pos += fw;
		}
	}
	
	
	public static void testIm2colInput() {

		ForkJoinPool forkJoinPool = new ForkJoinPool();
		int pad = 0;

		double[][][][] x = new double[][][][] {
			{
				{
					{1.1,1.2,1.3},
					{1.4,1.5,1.6},
					{1.7,1.8,1.9}
				},
				{
					{1.101,1.11,1.12},
					{1.13,1.14,1.15},
					{1.16,1.17,1.18}
				},
				{
					{1.19,1.201,1.21},
					{1.22,1.23,1.24},
					{1.25,1.26,1.27}
				}
			},
			{{{2.1,2.2,2.3},{2.4,2.5,2.6},{2.7,2.8,2.9}},{{2.101,2.11,2.12},{2.13,2.14,2.15},{2.16,2.17,2.18}},{{2.19,2.201,2.21},{2.22,2.23,2.24},{2.25,2.26,2.27}}},
			{{{3.1,3.2,3.3},{3.4,3.5,3.6},{3.7,3.8,3.9}},{{3.101,3.11,3.12},{3.13,3.14,3.15},{3.16,3.17,3.18}},{{3.19,3.201,3.21},{3.22,3.23,3.24},{3.25,3.26,3.27}}},
			{{{4.1,4.2,4.3},{4.4,4.5,4.6},{4.7,4.8,4.9}},{{4.101,4.11,4.12},{4.13,4.14,4.15},{4.16,4.17,4.18}},{{4.19,4.201,4.21},{4.22,4.23,4.24},{4.25,4.26,4.27}}},
			};

		int N = x.length;
		int C = x[0].length;
		int H = x[0][0].length;
		int W = x[0][0][0].length;

		int stride = 1;

		int fh = 2;
		int fw = 2;

		int oh = (H + 2 * pad - fh) / stride + 1;
		int ow = (W + 2 * pad - fw) / stride + 1;

		double[][] col = new double[x.length * oh * ow][fh * fw * C];

		Im2col im2col = new Im2col(x, fh, fw, oh, ow, stride, pad, col, 0, N - 1);
		ForkJoinTask<Void> a = forkJoinPool.submit(im2col);

		a.join();
		
		PrintUtils.printImage(col);

	}
	
	public static void testIm2colKeral() {
		
		double[][][][] x = {
				{
					{
						{11,11},
						{11,11}
					},
					{
						{12,12},
						{12,12}
					},
					{
						{13,13},
						{13,13}
					}
				},
				{
					{
						{21,21},
						{21,21}
					},
					{
						{22,22},
						{22,22}
					},
					{
						{23,23},
						{23,23}
					}
				},
				{
					{
						{31,31},
						{31,31}
					},
					{
						{32,32},
						{32,32}
					},
					{
						{33,33},
						{33,33}
					}
				},
				{
					{
						{41,41},
						{41,41}
					},
					{
						{42,42},
						{42,42}
					},
					{
						{43,43},
						{43,43}
					}
				}
				
		};
		
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		
		int fh = x[0][0].length;
		int fw = x[0][0][0].length;
		
		double[][] col = new double[x.length][fh * fw * x[0].length];
		
		Im2col im2col = new Im2col(x, fh , fw, 1, 1, 0, 0, col, 0, x.length - 1);
		
		ForkJoinTask<Void> a = forkJoinPool.submit(im2col);

		a.join();
		
		PrintUtils.printImage(col);
		
		double[][] t = MatrixUtils.transpose(col);
		
		System.out.println("==================T==========================");
		
		PrintUtils.printImage(t);
		
//		
//		double[][] transposedCol = new double[col[0].length][col.length];
//		
//		MatrixTransposion mt = new MatrixTransposion(col,transposedCol,0,col.length - 1);
//		
//		a = forkJoinPool.submit(mt);
//		
//		a.join();
		
		
		
	}
	
	
	private static  double[] toRow(double[][][][] x) {
		assert x != null;
		double[] result = new double[x.length * x[0].length * x[0][0].length * x[0][0][0].length];

		int nSize = x[0].length * x[0][0].length * x[0][0][0].length;
		int cSize = x[0][0].length * x[0][0][0].length;
		int wSize = x[0][0][0].length;
		
		for (int n = 0; n < x.length; n++)
			for (int c = 0; c < x[0].length; c++)
				for (int i = 0; i < x[0][0].length; i++) {
						int pos = n * nSize + c * cSize + i * wSize;
						System.arraycopy(x[n][c][i], 0, result, pos, wSize);
					}

		return result;
	}
	
	private static double[][] to2DimenArray(double[] x, int n, int d){
				
		double[][] result = new double[n][d];
		for(int i = 0; i < n; i ++) {
			System.arraycopy(x, i * d, result[i], 0, d);
		}
		
		return result;
	}
	
	public static double[][][][] to4DimenArray(double[] x, int N, int C, int H, int W){
		
		double[][][][] result = new double[N][C][H][W];
		
		double[][] mat = to2DimenArray(x, N * H * W, C);
		
		for(int n = 0;n<N;n++) {
			
			for(int c = 0;c<C;c++) {
			
				for(int h = 0;h<H;h++) {
					
					for(int w = 0;w<W;w++) {
						
						result[n][c][h][w] = mat[n * H * W + h * w + w][c];
						
					}
					
				}

			}
			
		}
		
		return result;
	}
	
	/**
	 * 把四维数组转换成二维数组
	 * @param x
	 * @param n
	 * @param d
	 * @return
	 */
	public static double[][] reshape(double[][][][] x, int n, int d) {
		assert x != null;
		assert n != 0 && d != 0;
		
		int nSize = x.length * x[0].length * x[0][0].length * x[0][0][0].length;
		
		if(n == -1) {
			n = nSize / d;
		}
		if(d == -1) {
			d = nSize / n;
		}
		
		assert nSize >= n * d;
		
		double[] row = toRow(x);
		
		double[][] result = to2DimenArray(row,n,d);
		
		return result;
	}
	
	public static void main(String[] args) {
		
		Im2col.testIm2colInput();
		
		System.out.println("=======================================>");
		
		Im2col.testIm2colKeral();
		
	}
	
}
