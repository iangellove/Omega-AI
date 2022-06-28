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
	
	public BNType bnType = BNType.fully_bn;
	
	public float[] mean;
	
	public float[] var;
	
	public float[] std;
	
	private float[] runingMean;

	private float[] runingStd;
	
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
	
	private float eta = 0.00000001f;
	
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
			this.channel = preLayer.oChannel;
			this.height = preLayer.oHeight;
			this.width = preLayer.oWidth;
			this.oChannel = this.channel;
			this.oHeight = this.height;
			this.oWidth = this.width;
		}
		if(this.preLayer.getLayerType() == LayerType.conv) {
			this.setBnType(BNType.conv_bn);
			this.meanNum = this.channel;
		}else if(this.preLayer.getLayerType() == LayerType.full){
			this.setBnType(BNType.fully_bn);
			this.meanNum = this.channel * this.height * this.width;
		}
		if(this.gama == null || this.beta == null) {
			this.gama = MatrixUtils.one(this.meanNum);
			this.beta = MatrixUtils.zero(this.meanNum);
			this.mean = MatrixUtils.zero(this.meanNum);
			this.var = MatrixUtils.zero(this.meanNum);
			this.std = MatrixUtils.zero(this.meanNum);
			this.deltaGama = MatrixUtils.zero(this.meanNum);
			this.deltaBeta = MatrixUtils.zero(this.meanNum);
		}
		this.number = this.network.number;
		this.output = Blobs.zero(number, oChannel, oHeight, oWidth, this.output);
		this.z = Blobs.zero(this.number, this.oChannel, this.oHeight, this.oWidth, this.z);
		MatrixUtils.zero(this.mean);
		MatrixUtils.zero(this.var);
		MatrixUtils.zero(this.std);
	}
	
	@Override
	public void initParam() {
		// TODO Auto-generated method stub
	}
	
	@Override
	public void initBack() {
		this.diff = Blobs.zero(number, channel, height, width, this.diff);
		MatrixUtils.zero(this.deltaGama);
		MatrixUtils.zero(this.deltaBeta);
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
			 MatrixOperation.mean(this.input.maxtir, this.mean, mode);

			/**
			 * 计算标准差
			 * var = 1/m ∑(x - mean)^2
			 * std = (var + eta)^1/2
			 */
			
			MatrixOperation.var(this.input.maxtir, this.mean, this.var, mode);
			
			MatrixOperation.std(MatrixOperation.add(this.var, this.eta), this.std);
			
			/**
			 * 移动加权平均法计算均值与方差
			 */
			this.runingMean = MWAUtils.mwa(this.mean, this.runingMean);
			
			this.runingStd = MWAUtils.mwa(this.std, this.runingStd);
			
			/**
			 * zi = (xi - mean) / (std + eta)
			 */
//			this.z.maxtir = MatrixOperation.division(MatrixOperation.subtraction(this.input.maxtir, this.mean, mode), this.std, mode);
			
			this.culOutput(this.mean, this.std, mode);
			
		}else {

			/**
			 * zi = (xi - mean) / (std + eta)
			 */
//			this.z.maxtir = MatrixOperation.division(MatrixOperation.subtraction(this.input.maxtir, this.runingMean, mode), this.runingStd, mode);
			
			this.culOutput(this.runingMean, this.runingStd, mode);
			
		}

		/**
		 * yi = gama * zi + beta
		 */
//		this.output.maxtir = MatrixOperation.addByBN(MatrixOperation.multiplicationByBN(this.z.maxtir, this.gama, mode), this.beta, mode);
		
	}
	
	/**
	 * zi = (xi - mean) / (std + eta)
	 * yi = gama * zi + beta
	 */
	private void culOutput(float[] m,float[] s,int mode) {
		
		int N = this.input.maxtir.length;
		int C = this.input.maxtir[0].length;
		int H = this.input.maxtir[0][0].length;
		int W = this.input.maxtir[0][0][0].length;

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
									z.maxtir[index][c][h][w] = (input.maxtir[index][c][h][w] - m[w]) / s[w];
									output.maxtir[index][c][h][w] = z.maxtir[index][c][h][w] * gama[w] + beta[w];
								}else {
									z.maxtir[index][c][h][w] = (input.maxtir[index][c][h][w] - m[c]) / s[c];
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

		/**
		 * deltaGama = ∑ deta * z
		 * deltaBeta = ∑ deta
		 * dxhat = deta * gama
		 * dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
		 * dmean = ∑ dxhat * -1 / (var + eta)^1/2 + dvar * (∑ -2 * (x - mean)) / n
		 * dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/2
		 */
//		float[][][][] dz = new float[this.number][this.channel][this.oHeight][this.oWidth];
		
//		long start = System.nanoTime();

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
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		for(int m = 0;m<this.number;m++) {
			final int index = m;
			workers.add(new Task<Object>(index) {
				@Override
			    public Object call() throws Exception {
					for(int c = 0;c<oChannel;c++) {
						for(int h = 0;h<oHeight;h++) {
							for(int w = 0;w<oWidth;w++) {
								int ci = w;
								if(bnType == BNType.conv_bn) {
									ci = c;
								}
								// deltaGama = ∑ deta * z
								deltaGama[ci] += delta.maxtir[index][c][h][w] * z.maxtir[index][c][h][w];
								// deltaBeta = ∑ deta
								deltaBeta[ci] += delta.maxtir[index][c][h][w];
								// dxhat = deta * gama
								diff.maxtir[index][c][h][w] = delta.maxtir[index][c][h][w] * gama[ci];
							}
						}
					}
					return null;
				}
			});
		}

		TaskEngine.getInstance(this.network.getThreadNum()).dispatchTask(workers);

		float[] meanDzSum = meanDzSum(input.maxtir, diff.maxtir, mean, mode);

		for(int i = 0;i<meanDzSum.length;i++) {
			// dvar = ∑ dxhat * (xi - mean) * -1/2 * (var + eta)^-3/2
			dvar[i] = (float) (meanDzSum[i] * -0.5f * Math.pow(var[i] + eta, -1.5));
		}
		
		/**
		 * 原论文公式
		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2) + dvar * (∑ -2 * (x - mean)) / n
		 * 使用darknet公式
		 * dmean = (∑ dxhat * -1 / (var + eta)^1/2)
		 */
		 muSum(diff.maxtir, std, dmu, mode);

//		BNBackward job = new BNBackward(this, dvar, dmu, 0, number - 1);
//		
//		ForkJobEngine.run(job);
		
//		System.out.println((System.nanoTime() - start) / 1e6 + "ms.");
		
		/**
		 * dl/dx
		 */
		for(int m = 0;m<this.number;m++) {
			for(int c = 0;c<oChannel;c++) {
				for(int h = 0;h<oHeight;h++) {
					for(int w = 0;w<oWidth;w++) {
						int ci = w;
						if(bnType == BNType.conv_bn) {
							ci = c;
						}
						// dx = dxhat * 1 / (var + eta)^1/2 + dvar * 2(x - mean) / n + dmean * 1/n
						diff.maxtir[m][c][h][w] = diff.maxtir[m][c][h][w] / std[ci] + 2.0f * dvar[ci] * (input.maxtir[m][c][h][w] - mean[ci]) / batchNum + dmu[ci] / batchNum;
					}
				}
			}
		}
		
//		System.out.println((System.nanoTime() - start) / 1e6+"ms.========>");
		
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
		
		float[][][][] output = new float[this.number][this.oChannel][this.oHeight][this.oWidth];
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
		return output;
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
	
	public float[] meanSum(float[][][][] x,float[] mean,int mode) {
		
		float[] meanSum = new float[mean.length];
		
		if(mode == 0) {
			for(int i = 0;i<meanSum.length;i++) {
				for(int n = 0;n<x.length;n++) {
					meanSum[i] += x[n][0][0][i] - mean[i];
				}
			}
		}else {
			for(int i = 0;i<meanSum.length;i++) {
				for(int n = 0;n<x.length;n++) {
					for(int h = 0;h<x[0][0].length;h++) {
						for(int w = 0;w<x[0][0][h].length;w++) {
							meanSum[i] += x[n][i][h][w] - mean[i];
						}
					}
				}
			}
		}

		return meanSum;
	}
	
	public float[] meanDzSum(float[][][][] x,float[][][][] dz,float[] mean,int mode) {
		
		float[] meanDzSum = new float[mean.length];
		
		if(mode == 0) {
			for(int i = 0;i<meanDzSum.length;i++) {
				for(int n = 0;n<x.length;n++) {
					meanDzSum[i] += (x[n][0][0][i] - mean[i]) * dz[n][0][0][i];
				}
			}
		}else {
			for(int i = 0;i<meanDzSum.length;i++) {
				for(int n = 0;n<x.length;n++) {
					for(int h = 0;h<x[0][0].length;h++) {
						for(int w = 0;w<x[0][0][h].length;w++) {
							meanDzSum[i] += (x[n][i][h][w] - mean[i]) * dz[n][i][h][w];
						}
					}
				}
			}
		}

		return meanDzSum;
	}
	
	public void muSum(float[][][][] dz,float[] std,float[] result,int mode) {
		
		if(mode == 0) {
			for(int i = 0;i<std.length;i++) {
				for(int n = 0;n<dz.length;n++) {
					result[i] += -1.0f * dz[n][0][0][i] / std[i];
				}
			}
		}else {
			for(int i = 0;i<std.length;i++) {
				for(int n = 0;n<dz.length;n++) {
					for(int h = 0;h<dz[0][0].length;h++) {
						for(int w = 0;w<dz[0][0][h].length;w++) {
							result[i] += -1.0f * dz[n][i][h][w] / std[i];
						}
					}
				}
			}
		}

	}
	
}
