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
import com.omega.engine.nn.layer.ParamsInit;
import com.omega.engine.nn.layer.PoolingLayer;
import com.omega.engine.nn.layer.RouteLayer;
import com.omega.engine.nn.layer.UPSampleLayer;
import com.omega.engine.nn.layer.YoloLayer;
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
		
		for(int i = 0;i<layerCfgs.size();i++) {
			
			Map<String,Object> cfg = layerCfgs.get(i);
			
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
			case "route":
				addRouteLayer(cfg, nn, layerCfgs, i);
				break;
			case "upsample":
				addUpsampleLayer(cfg, nn);
				break;
			case "yolo":
				addYoloLayer(cfg, nn);
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
	
	public static float getFloat(String val) {
		return new Float(val);
	}
	
	public static void addInputLayer(Map<String,Object> cfg,Network nn) {

		int channel = getInt(cfg.get("channel").toString());
		int width = getInt(cfg.get("width").toString());
		int height = getInt(cfg.get("height").toString());
		
		InputLayer inputLayer = new InputLayer(channel, height, width);

		nn.addLayer(inputLayer);
		cfg.put("lastIndex", inputLayer.index);
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
		
		cfg.put("lastIndex", pool1.index);
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
		
		cfg.put("lastIndex", pool1.index);
	}
	
	public static void addAvgPoolingLayer(Map<String,Object> cfg,Network nn) {
		
		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the pooling layer cant be the fisrt layer.");
		}
		
		AVGPoolingLayer pool1 = new AVGPoolingLayer(pre.oChannel, pre.oWidth, pre.oHeight);
		
		nn.addLayer(pool1);
		
		cfg.put("lastIndex", pool1.index);
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
		cfg.put("lastIndex", conv.index);
		
		if(bn == 1) {
			BNLayer bn1 = new BNLayer(conv);
			nn.addLayer(bn1);
			cfg.put("lastIndex", bn1.index);
		}
		
		Layer activeLayer = makeActivation(activation, conv);
		
		if(activeLayer != null) {
			nn.addLayer(activeLayer);
			cfg.put("lastIndex", activeLayer.index);
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
		cfg.put("lastIndex", fully.index);
		
		if(bn == 1) {
			BNLayer bn1 = new BNLayer();
			nn.addLayer(bn1);
			cfg.put("lastIndex", bn1.index);
		}
		
		Layer activeLayer = makeActivation(activation, fully);
		
		if(activeLayer != null) {
			nn.addLayer(activeLayer);
			cfg.put("lastIndex", activeLayer.index);
		}
		
	}
	
	public static void addRouteLayer(Map<String,Object> cfg,Network nn,List<Map<String,Object>> layerCfgs,int current) {
		
		List<Double> layerIndexList = (List<Double>) cfg.get("layers");
		int[] layerIndexs = new int[layerIndexList.size()];
		for(int i = 0;i<layerIndexList.size();i++) {
			layerIndexs[i] = layerIndexList.get(i).intValue();
		}
		
		Layer[] layers = new Layer[layerIndexs.length];
		
		for(int i = 0;i<layerIndexs.length;i++) {
			int ridx = layerIndexs[i];
			int index = 0;
			if(ridx < 0) {
				index = (int) layerCfgs.get(current + ridx).get("lastIndex");
			}else {
				index = (int) layerCfgs.get(ridx).get("lastIndex");
			}
			layers[i] = nn.layerList.get(index);
		}

		RouteLayer routeLayer = new RouteLayer(layers);
		System.out.println(routeLayer.oWidth);
		nn.addLayer(routeLayer);
		cfg.put("lastIndex", routeLayer.index);
	}
	
	public static void addUpsampleLayer(Map<String,Object> cfg,Network nn) {

		Layer pre = nn.getLastLayer();
		
		if(pre == null) {
			throw new RuntimeException("the upsample layer cant be the fisrt layer.");
		}
		
		int stride = getInt(cfg.get("stride").toString());
		
		UPSampleLayer upsampleLayer = new UPSampleLayer(pre.oChannel, pre.oHeight, pre.oWidth, stride);
		System.out.println(upsampleLayer.oWidth);
		nn.addLayer(upsampleLayer);
		cfg.put("lastIndex", upsampleLayer.index);
	}
	
	public static void addYoloLayer(Map<String,Object> cfg,Network nn) {

		int class_number = getInt(cfg.get("classes").toString());
		int total = getInt(cfg.get("num").toString());
		int maxBox = getInt(cfg.get("maxBox").toString());
		float ignoreThresh = getFloat(cfg.get("ignore_thresh").toString());
		float truthThresh = getFloat(cfg.get("truth_thresh").toString());
		
		List<Double> anchorsList = (List<Double>) cfg.get("anchors");
		float[] anchors = new float[anchorsList.size()];
		for(int i = 0;i<anchorsList.size();i++) {
			anchors[i] = anchorsList.get(i).floatValue();
		}
		
		List<Double> maskList = (List<Double>) cfg.get("mask");
		int[] mask = null;
		if(maskList != null) {
			mask = new int[maskList.size()];
			for(int i = 0;i<maskList.size();i++) {
				mask[i] = maskList.get(i).intValue();
			}
		}else {
			mask = new int[total];
		}
		
		YoloLayer yoloLayer = new YoloLayer(class_number, mask.length, mask, anchors, maxBox, total, ignoreThresh, truthThresh);
		nn.addLayer(yoloLayer);
		cfg.put("lastIndex", yoloLayer.index);
	}
	
	public static Layer makeActivation(String activation,Layer preLayer) {
		
		Layer layer = null;
		
		switch (activation) {
		case "relu":
			layer = new ReluLayer(preLayer);
			preLayer.paramsInit = ParamsInit.relu;
			break;
		case "sigmod":
			layer = new SigmodLayer(preLayer);
			preLayer.paramsInit = ParamsInit.sigmoid;
			break;
		case "leaky":
			layer = new LeakyReluLayer(preLayer);
			preLayer.paramsInit = ParamsInit.leaky_relu;
			break;
		case "tanh":
//			layer = new TanhLayer();
			preLayer.paramsInit = ParamsInit.tanh;
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
