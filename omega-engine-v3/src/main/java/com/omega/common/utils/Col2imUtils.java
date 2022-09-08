package com.omega.common.utils;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;
import com.omega.engine.gpu.GPUOP;

public class Col2imUtils extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2602908298290653960L;
	
	private int start;
	
	private int end;
	
	private float[] x;
	
	private float[] y;
	
	private int height;
	
	private int width;
	
	private int kh;
	
	private int kw;
	
	private int p;
	
	private int stride;
	
	private int oh;
	
	private int ow;
	
	public Col2imUtils(float[] x,float[] y,int height,int width,int kh,int kw,int p,int stride,int oh,int ow,int start,int end) {
		this.x = x;
		this.y = y;
		this.height = height;
		this.width = width;
		this.kh = kh;
		this.kw = kw;
		this.p = p;
		this.stride = stride;
		this.oh = oh;
		this.ow = ow;
		this.start = start;
		this.end = end;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		
		int length = end - start + 1;
		
		if (length < 8 || length <= y.length / 8) {
			
			col2();

		} else {

			int mid = (start + end + 1) >>> 1;
			Col2imUtils left = new Col2imUtils(x, y, height, width, kh, kw, p, stride, oh, ow, start, mid - 1);
			Col2imUtils right = new Col2imUtils(x, y, height, width, kh, kw, p, stride, oh, ow, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
		
	}
	
	private void col() {
		
		for (int index = start; index <= end; index++) {
			float val = 0;
	        int w = index % width + p;
	        int h = (index / width) % height + p;
	        int c = index / (width * height);
	        // compute the start and end of the output
	        int w_col_start = (w < kw) ? 0 : (w - kw) / stride + 1;
	        int w_col_end = Math.min(w / stride + 1, ow);
	        int h_col_start = (h < kh) ? 0 : (h - kh) / stride + 1;
	        int h_col_end = Math.min(h / stride + 1, oh);
	        // equivalent implementation
	        int offset = (c * oh * kw + h * kw + w) * oh * ow;
	        int coeff_h_col = (1 - stride * kh * oh) * ow;
	        int coeff_w_col = (1 - stride * oh * ow);
	        
	        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
	            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
	            	
	                val += x[offset + h_col * coeff_h_col + w_col * coeff_w_col];
	                
	            }
	        }
	        y[index] += val;	
		}
		
	}
	
	public void col2() {
		for (int index = start; index <= end; index++) {
		    float val = 0;
		    int w_im = index % width + p;
		    int h_im = (index / width) % height + p;
		    int c_im = index / (width * height);
		    // compute the start and end of the output
		    int w_col_start = (w_im < kw) ? 0 : (w_im - kw) / stride + 1;
		    int w_col_end = Math.min(w_im / stride + 1, ow);
		    int h_col_start = (h_im < kh) ? 0 : (h_im - kh) / stride + 1;
		    int h_col_end = Math.min(h_im / stride + 1, oh);
		    
		    // TODO: use LCM of stride and dilation to avoid unnecessary loops
		    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
		      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
		        int h_k = (h_im - h_col * stride);
		        int w_k = (w_im - w_col * stride);
		        int data_col_index = (((c_im * kh + h_k) * kw + w_k) * oh + h_col) * ow + w_col;
		        val += x[data_col_index];
		      }
		    }
		    y[index] = val;
		}
	}
	
	public static void main(String[] args) {

		int N = 2;
		int C = 3;
		int H = 4;
		int W = 4;
		int ko = 3;
		int kh = 3;
		int kw = 3;
		int p = 1;
		int s = 2;
		int oh = (H + 2 * p - kh) / s + 1;
		int ow = (W + 2 * p - kw) / s + 1;
		
		float[] delta = MatrixUtils.order(N * ko * oh * ow, 1);
		
		float[] kernel = MatrixUtils.order(ko * C * kh * kw, 1);
		
		float[] n_diff = new float[N * C * H * W];

		int m = C * kh * kw;
		int n = oh * ow;
		int k = ko;

		float[] r1 = new float[m * n];
		
		float[] out = new float[C * H * W];

		float[] onceDelta = new float[ko * oh * ow];

		for(int ni = 0;ni<N;ni++) {

			System.arraycopy(delta, ni * onceDelta.length, onceDelta, 0, onceDelta.length);
			
			GPUOP.getInstance().multiplyFloat(m, n, k, kernel, onceDelta, r1, 1, 0, 1.0f, 0.0f, m, n, n);

			Col2imUtils col2im = new Col2imUtils(r1, out, H, W, kh, kw, p, s, oh, ow, 0, out.length - 1);
			ForkJobEngine.run(col2im);
			
			System.arraycopy(out, 0, n_diff, ni * out.length, out.length);
			
		}
		
		System.out.println(JsonUtils.toJson(n_diff));
		
		int adj = (H + 2 * p - kh) % s;
		
		float[][][][] deltaADJ = MatrixUtils.transform(delta, N, C, oh, ow);
		
		int dh = oh + (oh - 1) * (s - 1);
		int dw = ow + (ow - 1) * (s - 1);
		float[][][][] dwd = new float[N][C][dh][dw];
		
		Dilation.dilation(deltaADJ, dwd, s);
		
		float[][][][] deltaP = null;
		
		if(s > 1) {
			int ppi = kh - p -1;

			deltaP = MatrixOperation.zeroPadding(dwd, ppi, adj);
		}
		
		float[] input1d = Im2colToVector.im2col(deltaP, kh, kw, 1);

		float[] r = new float[N * C * H * W];
		
		float[][][][] kernel180 = MatrixOperation.rotate180V2(MatrixUtils.transform(kernel, ko, C, kh, kw));
		
		float[] kt = Im2colUtils.kernalToVector(kernel180, true);
		
		int xm = N * H * W;
		int xn = kh * kw * ko;
		
		GPUOP.getInstance().multiplyFloat(xm, xn, C, input1d, kt, r);
		
		System.out.println("====================");
		
		System.out.println(JsonUtils.toJson(r));
		
		float[][][][] diff = new float[N][C][H][W];
		
		OP1dto4d.to1d(r, diff, N, C, H, W);
		
		float[] t2 = MatrixUtils.transform(diff);
		
		System.out.println(JsonUtils.toJson(t2));
		
		System.out.println(CheckArrayUtils.check(t2, n_diff));
		
		
	}
	
}
