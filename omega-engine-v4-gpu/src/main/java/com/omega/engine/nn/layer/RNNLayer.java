package com.omega.engine.nn.layer;

import com.omega.common.data.Tensor;
import com.omega.engine.gpu.GPUOP;
import com.omega.engine.nn.layer.active.ActiveFunctionLayer;
import com.omega.engine.nn.layer.gpu.RNNKernel;

import jcuda.Sizeof;
import jcuda.jcublas.cublasOperation;

/**
 * Recurrent Layer
 * @author Administrator
 *
 */
public class RNNLayer extends Layer{
	
	private int time = 0;
	
	private Tensor weight_u;
	
	private Tensor weight_w;
	
	private Tensor weight_v;
	
	private Tensor bias_u;
	
	private Tensor bias_v;
	
	private Tensor pre_z;
	
	private Tensor zt;
	
	private Tensor o1;
	
	private Tensor o2;
	
	private RNNKernel kernel;
	
	private ActiveFunctionLayer a1;
	
	@Override
	public void init() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initBack() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		/**
		 * ht = f(W * ht-1 + U * xt + bh)
		 * yt = f(V * ht + by)
		 */
		if(this.input != null) {
			
			for(int t = 0;t<time;t++) {
				
				GPUOP.getInstance().multiplyFloat(number, oWidth, width, input.getGpuData().withByteOffset(t * Sizeof.FLOAT), weight_u.getGpuData(), o1.getGpuData(),
						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
				
				GPUOP.getInstance().multiplyFloat(number, oWidth, width, pre_z.getGpuData(), weight_w.getGpuData(), o2.getGpuData(),
						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f); 
				
				if(hasBias) {
					kernel.addOutputBias(o1, o2, bias_u);
				}else {
					kernel.addOutput(o1, o2);
				}
				
				a1.forward(o1);
				
				pre_z = a1.output;
				
				GPUOP.getInstance().multiplyFloat(number, oWidth, width, a1.output.getGpuData(), weight_v.getGpuData(), output.getGpuData().withByteOffset(t * Sizeof.FLOAT),
						cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 1.0f, 0.0f);
				
				if(hasBias) {
					kernel.addBias(output, bias_v, t);
				}
				
			}
			
		}
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
		/**
		 * E = ∑et
		 * delta_u = ∑de/du
		 * delta_v = ∑de/dv
		 * delta_w = ∑de/dw
		 * 
		 */
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput();
		/**
		 * 计算输出
		 */
		this.output();
	}

	@Override
	public void back() {
		// TODO Auto-generated method stub
		
		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void forward(Tensor inpnut) {
		// TODO Auto-generated method stub
		
		/**
		 * 参数初始化
		 */
		this.init();
		/**
		 * 设置输入
		 */
		this.setInput(inpnut);
		/**
		 * 计算输出
		 */
		this.output();
		
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		this.initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();
		
		if(this.network.GRADIENT_CHECK) {
			this.gradientCheck();
		}

	}

	@Override
	public void update() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}

}
