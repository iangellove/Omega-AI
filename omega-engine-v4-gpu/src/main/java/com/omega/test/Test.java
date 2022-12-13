package com.omega.test;

import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.CheckArrayUtils;
import com.omega.common.utils.Im2colToVector;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixUtils;
import com.omega.common.utils.RandomUtils;
import com.omega.engine.gpu.Im2colKernel;
import com.omega.engine.gpu.JNITest;
import com.omega.engine.service.impl.BusinessServiceImpl;

public class Test {
	
	static {
		System.load("H:\\omega\\omega-ai\\omega-engine-v3\\jni\\test_cuda.dll");
	}
	
	public static void main(String[] args) {
		
//		new JNITest().max(1, 2);
		
//		BusinessServiceImpl bs = new BusinessServiceImpl();
//		bs.showImage();
//		bs.bpNetwork_iris();
//		bs.bpNetwork_mnist();
//		bs.cnnNetwork_mnist_demo();
//		bs.cnnNetwork_mnist();
//		bs.cnnNetwork_cifar10();
//		bs.cnnNetwork_vgg16_cifar10();
//		bs.vgg16_cifar10();
//		bs.alexNet_mnist();
//		bs.alexNet_cifar10();
		
//		int n = 3;
//		int x = 5;
//		
//		System.out.println((n + x) * 2);
//		
//		System.out.println(n * 2 + x * 2);
		
		int N = 128;
		int C = 512;
		int H = 34;
		int W = 34;
		int kH = 3;
		int kW = 3;
		int s = 1;
		
		int oHeight = ((H - kH ) / s) + 1;
		int oWidth = ((W - kW) / s) + 1;
		int ow = C * kH * kW;
		int oh = N * oHeight * oWidth;
		
		float[] x = RandomUtils.gaussianRandom(N * C * H * W, 0.1f);
		float[] y = new float[oh * ow];
		float[][][][] x2 = MatrixUtils.transform(x, N, C, H, W);
//		
//		JNITest f = new JNITest();
////
//		f.im2colV2(x, y, N, C, H, W, kH, kW, s);
//		
//		System.out.println(oh * ow);
//		
//		System.out.println(y[y.length - 1]);
//		
//		float[] y2 = new float[oh * ow];
//		
//		Im2colToVector.im2col(x2, y2, kH, kW, s);
//		
//		 System.out.println(CheckArrayUtils.check(y, y2));
		
//		long start = System.nanoTime();
//
////		Vector<Task<Object>> workers = new Vector<Task<Object>>();
//
//		for(int i = 0;i<10;i++) {
//			
//			JNITest.getInstance().im2colV2(x, y, N, C, H, W, kH, kW, s);
//			
//		}
//
////		TaskEngine.getInstance(8).dispatchTask(workers);
//
//    	System.out.println((System.nanoTime() - start) / 1e6 + "ms.1");
		
		float[] y2 = new float[oh * ow];

    	long start2 = System.nanoTime();

	    for(int i = 0;i<10;i++) {
	    	
	    	long start3 = System.nanoTime();
	    	
	    	Im2colToVector.im2col(x2, y2, kH, kW, s);
	    	
	    	System.out.println((System.nanoTime() - start3) / 1e6 + "ms.3");

    	}
	    
    	System.out.println((System.nanoTime() - start2) / 1e6 + "ms.2");
    	
//	    System.out.println(CheckArrayUtils.check(y, y2));
		
	}
	
}
