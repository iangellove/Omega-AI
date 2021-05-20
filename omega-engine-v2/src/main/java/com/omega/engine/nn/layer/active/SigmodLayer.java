package com.omega.engine.nn.layer.active;

import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.layer.LayerType;

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

	@Override
	public void output() {
		// TODO Auto-generated method stub
		for(int n = 0;n<this.number;n++) {
			for(int c = 0;c<this.channel;c++) {
				for(int h = 0;h<this.height;h++) {
					for(int w = 0;w<this.width;w++) {
						this.output.maxtir[n][c][h][w] = (double) (1d / (1d + Math.exp(-this.input.maxtir[n][c][h][w])));
					}
				}
			}
		}
	}

	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		for(int n = 0;n<this.number;n++) {
			for(int c = 0;c<this.channel;c++) {
				for(int h = 0;h<this.height;h++) {
					for(int w = 0;w<this.width;w++) {
						this.diff.maxtir[n][c][h][w] = this.delta.maxtir[n][c][h][w] * this.output.maxtir[n][c][h][w] * (1d - this.output.maxtir[n][c][h][w]);
					}
				}
			}
		}
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

}
