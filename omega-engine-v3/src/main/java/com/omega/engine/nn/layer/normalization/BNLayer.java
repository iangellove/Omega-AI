package com.omega.engine.nn.layer.normalization;

import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.MathUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;
import com.omega.engine.nn.data.Blob;
import com.omega.engine.nn.data.Blobs;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.model.LayerInit;
import com.omega.engine.nn.network.RunModel;
import com.omega.engine.updater.MWAUtils;

/**
 * 
 * Batch Normalization Layer
 * 
 * @author Administrator
 * 
 * mean = ∑x / m
 * std = (∑(x - mean)^2 / m)^1/2
 * zi = (xi - mean) / std
 * yi = gama * zi + beta
 */
public class BNLayer extends NormalizationLayer {
	
	public BNType bnType = null;
	
	public float[] mean;
	
	public float[] var;
	
//	public float[] std;
	
	private float[] runingMean;

	private float[] runingVar;
	
	/**
	 * if prelayer is conv layer meanNum = channel
	 * else if prelayer is fully layer meanNum = channel * height * width
	 */
	private int meanNum = 0;
	
//	
//	/**
//	 * std add eta square root
//	 * (std + eta)^1/2
//	 */
//	private float[][][] stdESR;
	
	public float[] gama;
	
	public float[] beta;
	
	private float eta = 0.00001f;
	
	/**
	 * zi = (xi - mean) / (std + eta)^1/2
	 */
	public Blob z;
	
	public float[] deltaGama;
	
	public float[] deltaBeta;

	public BNLayer() {
//		initParam();
	}
	
	@Override
	public void init() {
		if(preLayer == null) {
			preLayer = this.network.getPreLayer(this.index);
			if(this.parent != null) {
				preLayer = this.parent.layers.get(index - 1);
			}
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		
		if(this.bnType == null) {
			if(this.preLayer.getLayerType() == LayerType.conv) {
				this.setBnType(BNType.conv_bn);
				this.meanNum = this.channel;
			}else if(this.preLayer.getLayerType() == LayerType.full){
				this.setBnType(BNType.fully_bn);
				this.meanNum = this.channel * this.height * this.width;
			}
		}
		
		if(this.gama == null || this.beta == null) {
			this.gama = MatrixUtils.one(this.meanNum);
			this.beta = MatrixUtils.zero(this.meanNum);
			this.mean = MatrixUtils.zero(this.meanNum);
			this.var = MatrixUtils.zero(this.meanNum);
//			this.std = MatrixUtils.zero(this.meanNum);
			this.deltaGama = MatrixUtils.zero(this.meanNum);
			this.deltaBeta = MatrixUtils.zero(this.meanNum);
		}
		this.number = this.network.number;
		if(this.output == null || this.number != this.output.number) {
			this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
			this.z = Blobs.zero(this.number, this.oChannel, this.oHeight, this.oWidth, this.z);
		}
		MatrixUtils.zero(this.mean);
		MatrixUtils.zero(this.var);
//		MatrixUtils.zero(this.std);
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void initBack() {
		if(this.diff == null || this.number != this.diff.number) {
			this.diff = Blobs.zero(number, channel, height, width, this.diff);
		}
//		MatrixUtils.zero(this.deltaGama);
//		MatrixUtils.zero(this.deltaBeta);
	}

	@Override
	public void output() {
		// TODO Auto-generated method stub
		
		int mode = 0;
		
		switch (this.getBnType()) {
		case fully_bn:
			mode = 0;
			break;
		case conv_bn:
			mode = 1;
			break;
		}
		
		if(this.network.RUN_MODEL == RunModel.TRAIN) {
			
			
			/**
			 * 计算平均值
			 */
			MatrixOperation.meanV2(this.input.maxtir, this.mean, mode);
			
//			System.out.println(JsonUtils.toJson(mean));
			
//			 
//			float[] y = new float[this.mean.length];
//			
//			MatrixOperation.mean(this.input.maxtir, y, mode);
//
//			System.out.println(CheckArrayUtils.check(this.mean, y));
			
			/**
			 * 计算标准差
			 * var = 1/m ∑(x - mean)^2
			 * std = (var + eta)^1/2
			 */
			
//			MatrixOperation.var(this.input.maxtir, this.mean, this.var, mode);
			
			MatrixOperation.varV2(this.input.maxtir, this.mean, this.var, mode);
			
//			System.out.println(JsonUtils.toJson(var));
			
//			MatrixOperation.std(MatrixOperation.add(this.var, this.eta), this.std);
			
			/**
			 * 移动加权平均法计算均值与方差
			 */
			this.runingMean = MWAUtils.mwa(this.mean, this.runingMean);
			
			this.runingVar = MWAUtils.mwa(this.var, this.runingVar);
			
			/**
			 * zi = (xi - mean) / (std + eta)
			 */
//			this.z.maxtir = MatrixOperation.division(MatrixOperation.subtraction(this.input.maxtir, this.mean, mode), this.std, mode);
			
			this.culOutput(this.mean, this.var, mode);

		}else {

			/**
			 * zi = (xi - mean) / (std + eta)
			 */
//			this.z.maxtir = MatrixOperation.division(MatrixOperation.subtraction(this.input.maxtir, this.runingMean, mode), this.runingStd, mode);
			
			this.culOutput(this.runingMean, this.runingVar, mode);
			
		}
		
//		float[] t = MatrixUtils.transform(this.output.maxtir);
//		
//		System.out.println(JsonUtils.toJson(t));
		
		/**
		 * yi = gama * zi + beta
		 */
//		this.output.maxtir = MatrixOperation.addByBN(MatrixOperation.multiplicationByBN(this.z.maxtir, this.gama, mode), this.beta, mode);
//		
//		System.out.println("x:"+JsonUtils.toJson(MatrixUtils.transform(input.maxtir)));
//		
//		System.out.println("runingMean:"+JsonUtils.toJson(this.runingMean));
//		
//		System.out.println("runingVar:"+JsonUtils.toJson(this.runingVar));
//		
//		System.out.println("output:"+JsonUtils.toJson(MatrixUtils.transform(output.maxtir)));
//		
//		
//		System.out.println("gama:"+JsonUtils.toJson(gama));
//		
//		System.out.println("beta:"+JsonUtils.toJson(beta));
//		
		
	}
	
	/**
	 * zi = (xi - mean) / (std + eta)
	 * yi = gama * zi + beta
	 */
	private void culOutput(float[] m,float[] v,int mode) {
		
		int N = this.input.maxtir.length;
		int C = this.input.maxtir[0].length;
		int H = this.input.maxtir[0][0].length;
		int W = this.input.maxtir[0][0][0].length;

//		
//		System.out.println("m:"+JsonUtils.toJson(m));
//		System.out.println("v:"+JsonUtils.toJson(v));
//		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();

		for(int n = 0;n<N;n++) {
			final int index = n;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<C;c++) {
						for(int h = 0;h<H;h++) {
							for(int w = 0;w<W;w++) {
								
								if(mode == 0) {
									z.maxtir[index][c][h][w] = (input.maxtir[index][c][h][w] - m[w]) / (float) Math.sqrt(v[w] + eta);
									output.maxtir[index][c][h][w] = z.maxtir[index][c][h][w] * gama[w] + beta[w];
								}else {
									z.maxtir[index][c][h][w] = (input.maxtir[index][c][h][w] - m[c]) / (float) Math.sqrt(v[c] + eta);
									output.maxtir[index][c][h][w] = z.maxtir[index][c][h][w] * gama[c] + beta[c];
								}
	
							}
						}
					}
					return null;
				}
			});
			
		}
		
		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);
		
	}
	
	@Override
	public Blob getOutput() {
		// TODO Auto-generated method stub
		return this.output;
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

//	@Override
//	public void diff2() {
//		// TODO Auto-generated method stub
//
//		if(this.getBnType() == BNType.fully_bn) {
//			
//			/**
//			 * deltaGama = ∑ deta * z
//			 * deltaBeta = ∑ deta
//			 */
//			for(int m = 0;m<this.number;m++) {
//				for(int c = 0;c<this.oChannel;c++) {
//					for(int h = 0;h<this.oHeight;h++) {
//						for(int w = 0;w<this.oWidth;w++) {
//							this.deltaGama[w] += this.delta.maxtir[m][c][h][w] * this.z.maxtir[m][c][h][w] / this.number;
//							this.deltaBeta[w] += this.delta.maxtir[m][c][h][w] / this.number;
//						}
//					}
//				}
//			}
//
//			/**
//			 * 经过优化公式
//			 */
//			/**
//			 * 1/m * ∑ delta
//			 */
//			float[] meanD = MatrixOperation.mean(this.delta.maxtir, 0);
//			
//			/**
//			 * deltaXi = gama / std * (deltai - dgama * z / m - meanD)
//			 */
//			for(int m = 0;m<this.number;m++) {
//				for(int c = 0;c<this.oChannel;c++) {
//					for(int h = 0;h<this.oHeight;h++) {
//						for(int w = 0;w<this.oWidth;w++) {
//							diff.maxtir[m][c][h][w] = this.gama[w] / this.std[w] * (this.delta.maxtir[m][c][h][w] - this.deltaGama[w] * this.z.maxtir[m][c][h][w] / this.number - meanD[w]);
//						}
//					}
//				}
//			}
//			
//		}else {
//
//			/**
//			 * deltaGama = ∑ deta * z
//			 * deltaBeta = ∑ deta
//			 */
//			for(int m = 0;m<this.number;m++) {
//				for(int c = 0;c<this.oChannel;c++) {
//					for(int h = 0;h<this.oHeight;h++) {
//						for(int w = 0;w<this.oWidth;w++) {
//							this.deltaGama[c] += this.delta.maxtir[m][c][h][w] * this.z.maxtir[m][c][h][w] / this.number / this.oHeight / this.oWidth;
//							this.deltaBeta[c] += this.delta.maxtir[m][c][h][w] / this.number / this.oHeight / this.oWidth;
//						}
//					}
//				}
//			}
//
//			/**
//			 * 经过优化公式
//			 */
//			/**
//			 * 1/m * ∑ delta
//			 */
//			float[] meanD = MatrixOperation.mean(this.delta.maxtir, 1);
//			
//			/**
//			 * deltaXi = (gama / (var + eta)) * (deltai - dgama * z / m - meanD)
//			 */
//			for(int m = 0;m<this.number;m++) {
//				for(int c = 0;c<this.oChannel;c++) {
//					for(int h = 0;h<this.oHeight;h++) {
//						for(int w = 0;w<this.oWidth;w++) {
//							diff.maxtir[m][c][h][w] = this.gama[c] / this.std[c] * (this.delta.maxtir[m][c][h][w] - this.deltaGama[c] * this.z.maxtir[m][c][h][w] / this.number / this.oHeight / this.oWidth - meanD[c]);
//						}
//					}
//				}
//			}
//			
//		}
////		System.out.println(this.diff.maxtir[0][0][0][0]);
////		
////		MatrixOperation.printImage(this.diff.maxtir[0][0]);
////		
//	}

	/**
	 * 原论文公式
	 */
	@Override
	public void diff() {

//		long start = System.nanoTime();
		
		/**
		 * deltaGama = ∑ deta * z
		 * deltaBeta = ∑ deta
		 * dxhat = deta * gama
		 * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
		 * dmean = ∑ dxhat * -1 / (var + eta)^1/2 + dvar * (∑ -2 * (x - mean)) / n
		 * dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/2
		 */
//		float[][][][] dz = new float[this.number][this.channel][this.oHeight][this.oWidth];
		
		float[] dvar = new float[this.meanNum];

		float[] dmu = new float[this.meanNum];
		
		int batchNum = number;
		
		if(bnType == BNType.conv_bn) {
			batchNum = number * oHeight * oWidth;
		}
		
		int mode = 0;
		
		switch (this.getBnType()) {
		case fully_bn:
			mode = 0;
			break;
		case conv_bn:
			mode = 1;
			break;
		}
		
		computeDeltaV2();
		
		/**
		 * 原论文公式
		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
		 * 使用darknet公式
		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
		 */
		meanDzSum(input.maxtir, diff.maxtir, mean, var, dvar, dmu, mode);

		computeDiff(dvar, dmu, batchNum);

//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
		
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
		
		if(this.updater != null){
			this.updater.updateForBN(this);
		}else{
			for(int i = 0;i<this.gama.length;i++) {
				this.gama[i] -= this.learnRate * this.deltaGama[i];
			}
			for(int i = 0;i<this.beta.length;i++) {
				this.beta[i] -= this.learnRate * this.deltaBeta[i];
			}
		}
		
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub

		float[] x = MatrixUtils.transform(this.diff.maxtir);
		
		System.out.println("bn layer["+this.index+"]diff-max:"+MathUtils.max(x)+" min:"+MathUtils.min(x));
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.bn;
	}

	@Override
	public LayerInit save() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public float[][][][] output(float[][][][] input) {
		// TODO Auto-generated method stub
		
//		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
//		
//		/**
//		 * 计算平均值
//		 */
//		float[][][] mean = MatrixOperation.mean(input);
//		
//		/**
//		 * 计算标准差
//		 * var = 1/m ∑(x - mean)^2
//		 * std = (var + eta)^1/2
//		 */
//		float[][][] std = MatrixOperation.std(input);
//
//		float[][][] stdEta = MatrixOperation.add(std, this.eta);
//		
//		/**
//		 * zi = (xi - mean) / (std + eta)
//		 */
//		float[][][][] z = MatrixOperation.division(MatrixOperation.subtractionP(input, mean), stdEta);
//
//		/**
//		 * yi = gama * zi + beta
//		 */
//		output = MatrixOperation.addByBN(MatrixOperation.multiplicationByBN(z, this.gama), this.beta);
//		
		return this.output.maxtir;
	}
	
	public BNType getBnType() {
		return bnType;
	}

	public void setBnType(BNType bnType) {
		this.bnType = bnType;
	}

	@Override
	public void initCache() {
		// TODO Auto-generated method stub
		
	}
//	
//	public float[] meanSum(float[][][][] x,float[] mean,int mode) {
//		
//		float[] meanSum = new float[mean.length];
//		
//		if(mode == 0) {
//			for(int i = 0;i<meanSum.length;i++) {
//				for(int n = 0;n<x.length;n++) {
//					meanSum[i] += x[n][0][0][i] - mean[i];
//				}
//			}
//		}else {
//			for(int i = 0;i<meanSum.length;i++) {
//				for(int n = 0;n<x.length;n++) {
//					for(int h = 0;h<x[0][0].length;h++) {
//						for(int w = 0;w<x[0][0][h].length;w++) {
//							meanSum[i] += x[n][i][h][w] - mean[i];
//						}
//					}
//				}
//			}
//		}
//
//		return meanSum;
//	}
//	
//	public void meanDzSum(float[][][][] x,float[][][][] dz,float[] mean,float[] std,float[] dvar2,float[] dmu,float[] dmu2,float scale,int mode) {
//		
//		Vector<Task<Object>> workers = new Vector<Task<Object>>();
//		
//		if(mode == 0) {
//			for(int i = 0;i<dvar2.length;i++) {
//				final int index = i;
//				workers.add(new Task<Object>(index) {
//					@Override
//				    public Object call() throws Exception {
//						for(int n = 0;n<x.length;n++) {
//							dvar2[index] += (x[n][0][0][index] - mean[index]) * dz[n][0][0][index];
//							dmu[index] += -1.0f * dz[n][0][0][index] / std[index];
//							dmu2[index] += -2.0f * (x[n][0][0][index] - mean[index]) * scale;
//						}
//						return null;
//					}
//				});
//			}
//			
//		}else {
//			for(int i = 0;i<dvar2.length;i++) {
//				final int index = i;
//				workers.add(new Task<Object>(index) {
//					@Override
//				    public Object call() throws Exception {
//						for(int n = 0;n<x.length;n++) {
//							for(int h = 0;h<x[0][0].length;h++) {
//								for(int w = 0;w<x[0][0][h].length;w++) {
//									dvar2[index] += (x[n][index][h][w] - mean[index]) * dz[n][index][h][w];
//									dmu[index] += -1.0f * dz[n][index][h][w] / std[index];
//									dmu2[index] += -2.0f * (x[n][index][h][w] - mean[index]) * scale;
//								}
//							}
//						}
//						return null;
//					}
//				});
//				
//			}
//		}
//
//		TaskEngine.getInstance(8).dispatchTask(workers);
//		
//	}
	
	/**
	 * 原论文公式
	 * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
	 * 使用darknet公式
	 * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
	 */
	public void meanDzSum(float[][][][] x,float[][][][] dz,float[] mean,float[] var,float[] dvar,float[] dmu,int mode) {
		
		if(mode == 0) {
			for(int i = 0;i<dvar.length;i++) {
				float m_val = 0.0f;
				float v_val = 0.0f;
				for(int n = 0;n<x.length;n++) {
					m_val += dz[n][0][0][i];
					v_val += dz[n][0][0][i] * (x[n][0][0][i] - mean[i]);
				}
				dmu[i] = (float) (m_val * (-1.f/Math.sqrt(var[i] + eta)));
				dvar[i] = (float) (v_val * -0.5f * Math.pow(var[i] + eta, (float)(-3.0f/2.0f)));
			}
			
		}else {
			for(int i = 0;i<dvar.length;i++) {
				float m_val = 0.0f;
				float v_val = 0.0f;
				for(int n = 0;n<x.length;n++) {
					for(int h = 0;h<x[0][0].length;h++) {
						for(int w = 0;w<x[0][0][h].length;w++) {
							m_val += dz[n][i][h][w];
							v_val += dz[n][i][h][w] * (x[n][i][h][w] - mean[i]);
						}
					}
				}
				dmu[i] = (float) (m_val * (-1.f/Math.sqrt(var[i] + eta)));
				dvar[i] = (float) (v_val * -0.5f * Math.pow(var[i] + eta, (float)(-3.0f/2.0f)));
			}
		}

	}
	
	/**
	 * deltaGama = ∑ deta * z
	 * deltaBeta = ∑ deta
	 * dxhat = deta * gama
	 */
	public void computeDeltaV2() {
		
		if(bnType == BNType.conv_bn) {
			
			for(int c = 0;c<oChannel;c++) {
				deltaGama[c] = 0;
				deltaBeta[c] = 0;
				for(int m = 0;m<number;m++) {
					for(int h = 0;h<oHeight;h++) {
						for(int w = 0;w<oWidth;w++) {
							// deltaGama = ∑ deta * z
							deltaGama[c] += delta.maxtir[m][c][h][w] * z.maxtir[m][c][h][w];
							// deltaBeta = ∑ deta
							deltaBeta[c] += delta.maxtir[m][c][h][w];
							// dxhat = deta * gama
							diff.maxtir[m][c][h][w] = delta.maxtir[m][c][h][w] * gama[c];
						}
					}
				}

			}
			
		}else {

			for(int w = 0;w<oWidth;w++) {
				deltaGama[w] = 0;
				deltaBeta[w] = 0;
				for(int m = 0;m<number;m++) {
					// deltaGama = ∑ deta * z
					deltaGama[w] += delta.maxtir[m][0][0][w] * z.maxtir[m][0][0][w];
					// deltaBeta = ∑ deta
					deltaBeta[w] += delta.maxtir[m][0][0][w];
					// dxhat = deta * gama
					diff.maxtir[m][0][0][w] = delta.maxtir[m][0][0][w] * gama[w];

				}
			}

		}
		
	}
	
	/**
	 * dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/2
	 */
	public void computeDiff(float[] dvar,float[] dmu,float batchNum) {
		
		float sacle = 1.0f / batchNum;
		
		if(bnType == BNType.conv_bn) {

			/**
			 * dl/dx
			 */
			for(int m = 0;m<this.number;m++) {
				for(int c = 0;c<oChannel;c++) {
					for(int h = 0;h<oHeight;h++) {
						for(int w = 0;w<oWidth;w++) {
							// dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
							diff.maxtir[m][c][h][w] = diff.maxtir[m][c][h][w] / (float) Math.sqrt(var[c] + eta) + 2.0f * dvar[c] * (input.maxtir[m][c][h][w] - mean[c]) * sacle + dmu[c] * sacle;
						}
					}
				}
			}

		}else {

			for(int m = 0;m<this.number;m++) {
				for(int w = 0;w<oWidth;w++) {
					// dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
					diff.maxtir[m][0][0][w] = diff.maxtir[m][0][0][w] / (float) Math.sqrt(var[w] + eta) + 2.0f * dvar[w] * (input.maxtir[m][0][0][w] - mean[w]) * sacle + dmu[w] * sacle;
				}
			}

		}

	}

}
