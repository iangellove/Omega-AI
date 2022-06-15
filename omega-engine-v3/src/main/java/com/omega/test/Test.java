package com.omega.test;

import com.omega.engine.service.impl.BusinessServiceImpl;

public class Test {
	
	public static void main(String[] args) {
		
		BusinessServiceImpl bs = new BusinessServiceImpl();
//		bs.showImage();
//		bs.bpNetwork_iris();
//		bs.bpNetwork_mnist();
//		bs.cnnNetwork_mnist_demo();
//		bs.cnnNetwork_mnist();
//		bs.cnnNetwork_cifar10();
//		bs.cnnNetwork_vgg16_cifar10();
//		bs.vgg16_cifar10();
//		bs.alexNet_mnist();
		bs.alexNet_cifar10();
		
	}
	
}
