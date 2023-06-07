package com.omega.engine.gpu;

public class BaseGPUOP {
	
	private static BaseKernel kernel;
	
	public static BaseKernel getKernel() {
		if(kernel == null) {
			kernel = new BaseKernel();
		}
		return kernel;
	}

}
