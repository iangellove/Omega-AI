package com.omega.engine.updater.jobs;

import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class AdamJob extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5122995462148301836L;

	private float beta1 = 0.9f;
	
	private float beta2 = 0.999f;
	
	private float eta = 10e-8f;
	
	private float learnRate = 0.0001f;
	
	private int batch;
	
	private int time;
	
	private int start = 0;
	
	private int end = 0;
	
	private float[] diffW;
	
	private float[] mw;
	
	private float[] vw;
	
	private float[] weight;
	
	public AdamJob(float[] diffW,float[] mw,float[] vw,float[] weight,float learnRate,int batch,int time,int start,int end) {
		this.diffW = diffW;
		this.mw = mw;
		this.vw = vw;
		this.weight = weight;
		this.learnRate = learnRate;
		this.start = start;
		this.end = end;
		this.time = time;
		this.batch = batch;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = end - start + 1;
		
		if (length < 8 || length <= weight.length / 8) {
			
			exeute();

		} else {

			int mid = (start + end + 1) >>> 1;
			AdamJob left = new AdamJob(diffW, mw, vw, weight, learnRate, batch, time, start, mid - 1);
			AdamJob right = new AdamJob(diffW, mw, vw, weight, learnRate, batch, time, mid, end);

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}
	
	private void exeute() {
		
		for (int i = start; i <= end; i++) {
			
			mw[i] = beta1 * mw[i] + (1 - beta1) * diffW[i];
			vw[i] = beta2 * vw[i] + (1 - beta2) * diffW[i] * diffW[i];
			float mhat = (float) (mw[i] / (1 - Math.pow(beta1, time)));
			float vhat = (float) (vw[i] / (1 - Math.pow(beta2, time)));
			
			weight[i] = weight[i] - learnRate * mhat / ((float)Math.sqrt(vhat) + eta);
			
		}
		
	}
	
}
