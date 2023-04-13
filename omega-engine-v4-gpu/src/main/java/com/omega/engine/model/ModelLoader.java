package com.omega.engine.model;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.layer.AVGPoolingLayer;
import com.omega.engine.nn.layer.ConvolutionLayer;
import com.omega.engine.nn.layer.FullyLayer;
import com.omega.engine.nn.layer.InputLayer;
import com.omega.engine.nn.layer.Layer;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.active.LeakyReluLayer;
import com.omega.engine.nn.layer.active.ReluLayer;
import com.omega.engine.nn.layer.active.SigmodLayer;
import com.omega.engine.nn.layer.normalization.BNLayer;
import com.omega.engine.nn.network.Network;
import com.omega.engine.pooling.PoolingType;

/**
 * ModelLoader
 * @author Administrator
 *
 */
public class ModelLoader {
	
	public static void loadConfigToModel(Network nn,String filepath) {
		
		try {
			
			List<Map<String,Object>> layerCfgs = loadData(filepath);
			
			if(layerCfgs == null) {
				throw new RuntimeException("load the config file error.");
			}
			
			addLayer(layerCfgs, nn);
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	@SuppressWarnings("unused")
	public static void addLayer(List<Map<String,Object>> layerCfgs,Network nn) {
		
		for(Map<String,Object> cfg:layerCfgs) {
			
			String layerType = cfg.get("layerType").toString();
			
			Layer layer = null;
			
			switch (layerType) {
			case "fully":
				addFullyLayers(cfg, nn);
				break;
			case "convolutional":
				addConvLayers(cfg, nn);
				break;
			case "maxpool":
				addMaxPoolingLayer(cfg, nn);
				break;
			case "meanpool":
				addMeanPoolingLayer(cfg, nn);
				break;
			case "avgpool":
				addAvgPoolingLayer(cfg, nn);
				break;
			case "input":
				addInputLayer(cfg, nn);
				break;
			default:
				break;
			}
			System.out.println(layerType);
		}
		
	}
	
	public static int getInt(String val) {
		return new Double(val).intValue();
	}
	
	public static void addInputLayer(Map<String,Object> cfg,Network nn) {

		int channel = getInt(cfg.get("channel").toString());
		int width = getInt(cfg.get("width").toString());
		int height = getInt(cfg.get("height").toString());
		
		InputLayer inputLayer = new InputLayer(channel, height, width);

		nn.addLayer(inputLayer);
	}
	
	public static void addMaxPoolingLayer(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the pooling layer cant be the fisrt layer.");
		}
		
		int size = getInt(cfg.get("size").toString());
		int stride = getInt(cfg.get("stride").toString());
//		System.out.println(pre.oWidth);
		PoolingLayer pool1 = new PoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight, size, size, stride, PoolingType.MAX_POOLING);
		
		nn.addLayer(pool1);
	}
	
	public static void addMeanPoolingLayer(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the pooling layer cant be the fisrt layer.");
		}
		
		int size = getInt(cfg.get("size").toString());
		int stride = getInt(cfg.get("stride").toString());
//		System.out.println(pre.oWidth);
		PoolingLayer pool1 = new PoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight, size, size, stride, PoolingType.MEAN_POOLING);
		
		nn.addLayer(pool1);
	}
	
	public static void addAvgPoolingLayer(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the pooling layer cant be the fisrt layer.");
		}
		
		AVGPoolingLayer pool1 = new AVGPoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight);
		
		nn.addLayer(pool1);
	}
	
	public static void addConvLayers(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the convolution layer cant be the fisrt layer.");
		}
		
		int kernel = getInt(cfg.get("kernel").toString());
		int size = getInt(cfg.get("size").toString());
		int stride = getInt(cfg.get("stride").toString());
		int pad = getInt(cfg.get("pad").toString());
		
		int bn = 0;
		boolean hasBias = true;
		
		if(cfg.get("batch_normalize") != null) {
			bn = getInt(cfg.get("batch_normalize").toString());
			hasBias = false;
		}
		
		String activation = null;
		
		if(cfg.get("activation") != null) {
			activation = cfg.get("activation").toString();
		}

		ConvolutionLayer conv = new ConvolutionLayer(pre.oChannel, kernel, pre.oWidth, pre.oHeight, size, size, pad, stride, hasBias);
		System.out.println(conv.oWidth);
		nn.addLayer(conv);
		
		if(bn == 1) {
			BNLayer bn1 = new BNLayer(conv);
			nn.addLayer(bn1);
		}
		
		Layer activeLayer = makeActivation(activation, conv);
		
		if(activeLayer != null) {
			nn.addLayer(activeLayer);
		}
		
	}
	
	public static void addFullyLayers(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the fully layer cant be the fisrt layer.");
		}
		
		int inputSize = pre.oChannel * pre.oHeight * pre.oWidth;
		int outputSize = getInt(cfg.get("output").toString());
		
		int bn = 0;
		boolean hasBias = true;
		
		if(cfg.get("batch_normalize") != null && cfg.get("batch_normalize").toString().equals("1")) {
			bn = getInt(cfg.get("batch_normalize").toString());
			hasBias = false;
		}
		
		String activation = null;
		
		if(cfg.get("activation") != null) {
			activation = cfg.get("activation").toString();
		}
		System.out.println(inputSize);
		FullyLayer fully = new FullyLayer(inputSize, outputSize, hasBias);
		
		nn.addLayer(fully);
		
		if(bn == 1) {
			BNLayer bn1 = new BNLayer();
			nn.addLayer(bn1);
		}
		
		Layer activeLayer = makeActivation(activation, fully);
		
		if(activeLayer != null) {
			nn.addLayer(activeLayer);
		}
		
	}
	
	public static Layer makeActivation(String activation,Layer preLayer) {
		
		Layer layer = null;
		
		switch (activation) {
		case "relu":
			layer = new ReluLayer(preLayer);
			break;
		case "sigmod":
			layer = new SigmodLayer(preLayer);
			break;
		case "leaky":
			layer = new LeakyReluLayer(preLayer);
			break;
		case "tanh":
//			layer = new TanhLayer();
			break;
		}
		
		return layer;
	}
	
	@SuppressWarnings("unchecked")
	public static List<Map<String,Object>> loadData(String filepath){
		
		try {
			
			File file = new File(filepath);
			
			if(file.exists()) {
				
				try (
					FileInputStream fos = new FileInputStream(file);
					Reader reader = new InputStreamReader(fos, "utf-8");
					) {
					
					int ch = 0;
		            StringBuffer sb = new StringBuffer();
		            while ((ch = reader.read()) != -1) {
		                sb.append((char) ch);
		            }
		            
		            String json = sb.toString();
		            
		            List<Map<String,Object>> list = new ArrayList<Map<String,Object>>();
		            
		            list = JsonUtils.gson.fromJson(json, list.getClass());
		            
		            return list;
				} catch (Exception e) {
					// TODO: handle exception
					e.printStackTrace();
				}
				
			}else {
				throw new RuntimeException("the config file is not exists.");
			}

		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
}
