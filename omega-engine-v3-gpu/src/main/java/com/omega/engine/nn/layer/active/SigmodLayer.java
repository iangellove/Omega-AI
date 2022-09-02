package com.omega.engine.nn.layer.active;

import java.util.Vector;

import com.omega.common.data.Tensor;
import com.omega.common.task.ForkJobEngine;
import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.layer.active.jobs.sigmod.SigmodBackwardJob;
import com.omega.engine.nn.layer.active.jobs.sigmod.SigmodForwardJob;

/**
 * Sigmod active function Layer
 * @author Administrator
 *
 */
public class SigmodLayer extends ActiveFunctionLayer {

	@Override
	public void initParam() {
		// TODO Auto-generated method stub
		
	}

	public void init() {
		super.init();
	}
	
	@Override
	public void output() {
		// TODO Auto-generated method stub
		SigmodForwardJob forward = new SigmodForwardJob(input.data, output.data, 0, output.dataLength - 1);
		ForkJobEngine.run(forward);
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		SigmodBackwardJob backward = new SigmodBackwardJob(delta.data, output.data, diff.data, 0, diff.dataLength - 1);
		ForkJobEngine.run(backward);
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
		return LayerType.sigmod;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		
		for(int n = 0;n<this.number;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<channel;c++) {
						for(int h = 0;h<height;h++) {
							for(int w = 0;w<width;w++) {
								output[index][c][h][w] = (float) (1f / (1f + Math.exp(-input[index][c][h][w])));
							}
						}
					}
					return null;
				}
			});
		}
		
		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
		
		return output;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}


}
