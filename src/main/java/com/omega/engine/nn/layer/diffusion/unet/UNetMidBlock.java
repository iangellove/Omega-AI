package com.omega.engine.nn.layer.diffusion.unet;

import java.util.Map;

import com.omega.common.data.Tensor;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.LayerType;
import com.omega.engine.nn.network.Network;

/**
 * UNetDownBlock
 * @author Administrator
 *
 */
public class UNetMidBlock extends Layer{
	
	private int headNum = 4;
	
	private int contextTime = 0;
	
	private int contextDim = 0;
	
	private int nTime = 0;
	
	private int groupNum = 32;
	
	public UNetResidualBlock res_head;
	public UNetAttentionBlock attns;
	public UNetResidualBlock res_fail;
	
	public UNetMidBlock(int channel,int oChannel,int height,int width,int nTime,int headNum,int contextTime,int contextDim, int groupNum, Network network) {
		this.network = network;
		this.nTime = nTime;
		this.groupNum = groupNum;
		this.headNum = headNum;
		this.contextTime = contextTime;
		this.contextDim = contextDim;
		this.channel = channel;
		this.oChannel = oChannel;
		this.height = height;
		this.width = width;
		this.oHeight = height;
		this.oWidth = width;
		initLayers();
	}
	
	public void initLayers() {
		res_head = new UNetResidualBlock(channel, oChannel, height, width, nTime, groupNum, network);
		attns = new UNetAttentionBlock(oChannel, height, width, headNum, contextTime, contextDim, groupNum, network);
		res_fail = new UNetResidualBlock(oChannel, oChannel, height, width, nTime, groupNum, network);
	}

	@Override
	public void init() {
		// TODO Auto-generated method stub
		this.number = this.network.number;
	}
	
	public void init(Tensor input) {
		// TODO Auto-generated method stub
		this.number = input.number;
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

	}
	
	public void output(Tensor time) {
		// TODO Auto-generated method stub
		
		res_head.forward(input, time);
		attns.forward(res_head.getOutput());
		res_fail.forward(attns.getOutput(), time);
		
		this.output = res_fail.getOutput();
	}
	
	public void output(Tensor time,Tensor context) {
		// TODO Auto-generated method stub
		
		res_head.forward(input, time);
		attns.forward(res_head.getOutput(), context);
		res_fail.forward(attns.getOutput(), time);
		
		this.output = res_fail.getOutput();
	}

	@Override
	public Tensor getOutput() {
		// TODO Auto-generated method stub
		return this.output;
	}

	@Override
	public void diff() {
		// TODO Auto-generated method stub
		
	}
	
	public void diff(Tensor timeDiff) {
		// TODO Auto-generated method stub
		res_fail.back(delta, timeDiff);
		attns.back(res_fail.diff);
		res_head.back(attns.diff, timeDiff);
		
		this.diff = res_head.diff;
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

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta();
		/**
		 * 计算梯度
		 */
		this.diff();
	}

	@Override
	public void backTemp() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward(Tensor input) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output();
	}
	
	public void forward(Tensor input,Tensor time) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output(time);
	}
	
	public void forward(Tensor input,Tensor time,Tensor context) {
		// TODO Auto-generated method stub
		/**
		 * 参数初始化
		 */
		this.init(input);
		
		/**
		 * 设置输入
		 */
		this.setInput(input);

		/**
		 * 计算输出
		 */
		this.output(time, context);
	}

	@Override
	public void back(Tensor delta) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff();

	}
	
	public void back(Tensor delta,Tensor timeDiff) {
		// TODO Auto-generated method stub

		initBack();
		/**
		 * 设置梯度
		 */
		this.setDelta(delta);
		/**
		 * 计算梯度
		 */
		this.diff(timeDiff);

	}
	
	@Override
	public void update() {
		// TODO Auto-generated method stub
		res_head.update();
		attns.update();
		res_fail.update();
	}

	@Override
	public void showDiff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public LayerType getLayerType() {
		// TODO Auto-generated method stub
		return LayerType.unet_down;
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

	@Override
	public void accGrad(float scale) {
		// TODO Auto-generated method stub
		res_head.accGrad(scale);
		attns.accGrad(scale);
		res_fail.accGrad(scale);
	}

	public static void loadWeight(Map<String, Object> weightMap, UNetMidBlock network, boolean showLayers) {
		if(showLayers) {
			for(String key:weightMap.keySet()) {
				System.out.println(key);
			}
		}
//		
//		network.gn_feature.gamma = ClipModelUtils.loadData(network.gn_feature.gamma, weightMap, 1, "groupnorm_feature.weight");
//		network.gn_feature.beta = ClipModelUtils.loadData(network.gn_feature.beta, weightMap, 1, "groupnorm_feature.bias");
//		
//		ClipModelUtils.loadData(network.conv_feature.weight, weightMap, "conv_feature.weight");
//		ClipModelUtils.loadData(network.conv_feature.bias, weightMap, "conv_feature.bias");
//		
//		ClipModelUtils.loadData(network.temb.linear.weight, weightMap, "linear_time.weight");
//		ClipModelUtils.loadData(network.temb.linear.bias, weightMap, "linear_time.bias");
//		
//		network.gn_merged.gamma = ClipModelUtils.loadData(network.gn_merged.gamma, weightMap, 1, "groupnorm_merged.weight");
//		network.gn_merged.beta = ClipModelUtils.loadData(network.gn_merged.beta, weightMap, 1, "groupnorm_merged.bias");
//		
//		ClipModelUtils.loadData(network.conv_merged.weight, weightMap, "conv_merged.weight");
//		ClipModelUtils.loadData(network.conv_merged.bias, weightMap, "conv_merged.bias");
//		
//		network.residual_layer.weight = ClipModelUtils.loadData(network.residual_layer.weight, weightMap, 4, "residual_layer.weight");
//		ClipModelUtils.loadData(network.residual_layer.bias, weightMap, "residual_layer.bias");
		
	}
	
	public static void main(String[] args) {
		
	}
	
}
