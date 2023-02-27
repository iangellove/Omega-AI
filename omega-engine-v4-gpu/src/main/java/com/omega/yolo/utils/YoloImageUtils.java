package com.omega.yolo.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.OutputStreamWriter;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import com.omega.common.utils.ImageUtils;
import com.omega.common.utils.JsonUtils;
import com.omega.engine.nn.data.ImageData;
import com.omega.yolo.model.YoloImage;



public class YoloImageUtils {
	
	public static final String[] GL_CLASSES = new String[] {"person", "bird", "cat", "cow", "dog", "horse", "sheep",
	                                                         "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
	                                                         "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"};
	
	public static final int YOLO_IMG_SIZE = 448;
	
	public static final int GRID_SIZE = 7;
	
	private static ImageUtils iu;
	
	public static ImageUtils IU() {
		if(iu == null) {
			iu = new ImageUtils();
		}
		return iu;
	}
	
	public static int[] resize(String src,String outPath,int rw,int rh) throws Exception {
		
		ImageData data =  IU().getImageData(src);
		
		int padw = 0;
		int padh = 0;
		int w = data.getWeight();
		int h = data.getHeight();
		
		if(h > w) {
			padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}else if(h < w){
			padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}
		
		int ow = w + padw * 2;
		int oh = h + padh * 2;
		
		int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
		
		IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh);
		
		return new int[] {w, h, padw, padh};
	}
	
	public static int[] resize(ImageData data,String outPath,int rw,int rh,int padw,int padh,int w,int h) throws Exception {
		
		if(h > w) {
			padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}else if(h < w){
			padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}
		
		int ow = w + padw * 2;
		int oh = h + padh * 2;
		
		int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
		
		IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh);
		
		return square;
	}
	
	public static int[] resize(ImageData data,String outPath,int rw,int rh,int padw,int padh,int w,int h,int[][] bbox) throws Exception {
		
		if(h > w) {
			padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}else if(h < w){
			padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
		}
		
		int ow = w + padw * 2;
		int oh = h + padh * 2;
		
		int[] square = paddingToSquare(data.getColor(), w, h, padw, padh, ow, oh);
		
		IU().createRGBImage(outPath, data.getExtName(), ImageUtils.color2rgb(square, ow, oh), rw, rh, bbox);
		
		return square;
	}
	
	public static int[] paddingToSquare(int[] input,int w,int h,int padw,int padh,int ow,int oh){

		int[] output = new int[3 * ow * oh];
//		System.out.println(ow+":"+oh);
//		System.out.println(padw+":"+padh+":"+output.length);
		
		for(int x = 0;x<w;x++) {
			for(int y = 0;y<h;y++) {
				output[0 * ow * oh + (x + padw) * oh + (y + padh)] = input[0 * w * h + x * h + y];
				output[1 * ow * oh + (x + padw) * oh + (y + padh)] = input[1 * w * h + x * h + y];
				output[2 * ow * oh + (x + padw) * oh + (y + padh)] = input[2 * w * h + x * h + y];
			}
		}
		
		return output;
	}
	
	public static int[][] anno2bbox(String src){
		
		try {
			
			File file = new File(src);
			
			if(file.exists()) {

				Document doc = XmlParser.DB().parse(file);
				
				Element docEle = doc.getDocumentElement();
				
				NodeList objList = docEle.getElementsByTagName("object");
				
				if(objList == null || objList.getLength() <= 0) {
					return null;
				}
				
				int[][] output = new int[objList.getLength()][5];
				
				for(int i = 0;i<objList.getLength();i++){
					
					Node node = objList.item(i); 
					
					if(node.getNodeType() == Node.ELEMENT_NODE) {
						
						Element e = (Element) node;
						
						NodeList nodeList = e.getElementsByTagName("name");
						String name = nodeList.item(0).getChildNodes().item(0).getNodeValue();
						int nidx = getIndex(name);
//						System.out.println(nidx);
						if(nidx < 0) {
							continue;
						}
						
						nodeList = e.getElementsByTagName("bndbox");
						
						Element boxe = (Element) nodeList.item(0);
						
						nodeList = boxe.getElementsByTagName("xmin");
						int xmin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
						
						nodeList = boxe.getElementsByTagName("ymin");
						int ymin = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
						
						nodeList = boxe.getElementsByTagName("xmax");
						int xmax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
						
						nodeList = boxe.getElementsByTagName("ymax");
						int ymax = Integer.parseInt(nodeList.item(0).getChildNodes().item(0).getNodeValue());
						
						output[i][0] = nidx;
						output[i][1] = xmin;
						output[i][2] = ymin;
						output[i][3] = xmax;
						output[i][4] = ymax;
					}
					
				}
				
				return output;
			}
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static int getIndex(String name) {
		
		for(int i = 0;i<GL_CLASSES.length;i++) {
			if(name.equals(GL_CLASSES[i])) {
				return i;
			}
		}
		
		return -1;
	}
	
	/**
	 * w = wmax - wmin
	 * h = hmax - hmin
	 * cx = (wmax + wmin) / 2
	 * cy = (hmax + hmin) / 2
	 * @param label
	 * @return
	 */
	public static int[][] formatLabel(int[][] label) {
		
		int[][] bbox = new int[label.length][label[0].length];
		
		for(int i = 0;i<label.length;i++) {
			
			bbox[i][0] = label[i][0];
			int xmin = label[i][1];
			int ymin = label[i][2];
			int xmax = label[i][3];
			int ymax = label[i][4];
			int cx = (xmax + xmin) / 2;
			int cy = (ymax + ymin) / 2;
			int w = xmax - xmin;
			int h = ymax - ymin;
			bbox[i][1] = cx;
			bbox[i][2] = cy;
			bbox[i][3] = w;
			bbox[i][4] = h;
		}
		
		return bbox;
	}
	
	public static int[][] resizeBBox(int padw,int padh,int w,int h,int[][] bbox) {
		
		for(int i = 0;i<bbox.length;i++) {
			
			if(padw != 0) {
				bbox[i][1] = (bbox[i][1] + padw) * YOLO_IMG_SIZE / h;
				bbox[i][2] = (bbox[i][2]) * YOLO_IMG_SIZE / h;
				bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / h;
				bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / h;
			}
			
			if(padh != 0) {
				bbox[i][1] = (bbox[i][1]) * YOLO_IMG_SIZE / w;
				bbox[i][2] = (bbox[i][2] + padh) * YOLO_IMG_SIZE / w ;
				bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / w;
				bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / w;
			}
			
			if(padw == 0 && padh == 0) {
				bbox[i][1] = (bbox[i][1]) * YOLO_IMG_SIZE / w;
				bbox[i][2] = (bbox[i][2]) * YOLO_IMG_SIZE / w ;
				bbox[i][3] = (bbox[i][3]) * YOLO_IMG_SIZE / w;
				bbox[i][4] = (bbox[i][4]) * YOLO_IMG_SIZE / w;
			}
			
		}
		
		return bbox;
	}
	
	public static YoloImage formatData(String filename,String imgDir,String labelDir,String imgOutDir,boolean boxTest) {
		
		try {

			String dataName = filename.substring(0, filename.lastIndexOf("."));
			
			String imgPath = imgDir + "\\" + filename;
			String labelPath = labelDir + "\\" + dataName + ".xml";
			String imgOutPath = imgOutDir + "\\" + filename;
			String imgTestOutPath = imgOutDir + "\\test_" + filename;
			
			ImageData data =  IU().getImageData(imgPath);
			
			int padw = 0;
			int padh = 0;
			int w = data.getWeight();
			int h = data.getHeight();
			
			if(h > w) {
				padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}else if(h < w){
				padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}
			
			int[][] label = anno2bbox(labelPath);
			int[][] bbox = formatLabel(label);
			
			IU().createRGBImage(imgTestOutPath, "jpg",  ImageUtils.color2rgb(data.getColor(), w, h), bbox);

			bbox = resizeBBox(padw, padh, w, h, bbox);
			
			if(boxTest) {

				YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h, bbox);
				
			}else {

				YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
				
			}

//			System.out.println(JsonUtils.toJson(bbox));
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	} 
	
	public static YoloImage formatData(String filename,String imgDir,String labelDir,String imgOutDir) {
		
		try {
			
			YoloImage obj = new YoloImage();
			
			String dataName = filename.substring(0, filename.lastIndexOf("."));
			
			String imgPath = imgDir + "\\" + filename;
			String labelPath = labelDir + "\\" + dataName + ".xml";
			String imgOutPath = imgOutDir + "\\" + filename;
			
			ImageData data = IU().getImageData(imgPath);
			
			int padw = 0;
			int padh = 0;
			int w = data.getWeight();
			int h = data.getHeight();
			
			if(h > w) {
				padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}else if(h < w){
				padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}
			
			int[][] label = anno2bbox(labelPath);
			int[][] bbox = formatLabel(label);
			
			bbox = resizeBBox(padw, padh, w, h, bbox);
			
			int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
			System.out.println(dataName);
			obj.setName(dataName);
			obj.setChannel(3);
			obj.setHeight(YOLO_IMG_SIZE);
			obj.setWidth(YOLO_IMG_SIZE);
			obj.setBbox(bbox);
			obj.setData(resizeData);
			obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE));
			
			return obj;
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static YoloImage formatData(File file,String labelDir,String imgOutDir) {
		
		try {
			
			YoloImage obj = new YoloImage();
			
			String filename = file.getName();
			
			String dataName = filename.substring(0, filename.lastIndexOf("."));

			String labelPath = labelDir + "\\" + dataName + ".xml";
			String imgOutPath = imgOutDir + "\\" + filename;
			
			ImageData data =  IU().getImageData(file);
			
			int padw = 0;
			int padh = 0;
			int w = data.getWeight();
			int h = data.getHeight();
			
			if(h > w) {
				padw = new BigDecimal(h).subtract(new BigDecimal(w)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}else if(h < w){
				padh = new BigDecimal(w).subtract(new BigDecimal(h)).divide(new BigDecimal(2),BigDecimal.ROUND_DOWN).intValue();
			}
			
			int[][] label = anno2bbox(labelPath);
			int[][] bbox = formatLabel(label);
			
			bbox = resizeBBox(padw, padh, w, h, bbox);

			int[] resizeData = YoloImageUtils.resize(data, imgOutPath, YOLO_IMG_SIZE, YOLO_IMG_SIZE, padw, padh, w, h);
			
			obj.setName(dataName);
			obj.setChannel(3);
			obj.setHeight(YOLO_IMG_SIZE);
			obj.setWidth(YOLO_IMG_SIZE);
			obj.setBbox(bbox);
			obj.setData(resizeData);
//			System.out.println(dataName+":");
			obj.setYoloLabel(LabelUtils.labelToYolo(bbox, GRID_SIZE));
			
			return obj;
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
		return null;
	}
	
	public static void image2Yolo(String srcDir,String labelDir,String outDir,String bboxPath) {
		
		try {
			
			File file = new File(srcDir);
			
			Map<String, Object> bboxList = new HashMap<String, Object>(); 
			
			if(file.exists() && file.isDirectory()) {
				
				for(File img:file.listFiles()) {
					
					YoloImage yi = formatData(img, labelDir, outDir);
					System.out.println(yi.getName());
					bboxList.put(yi.getName(), yi.getYoloLabel());
					
				}
				
			}
			
			
			File txt = new File(bboxPath);
			
			if(!txt.exists()) {
				txt.createNewFile(); // 创建新文件,有同名的文件的话直接覆盖
			}
			
			try (FileOutputStream fos = new FileOutputStream(txt);
//					OutputStreamWriter osr = new OutputStreamWriter(fos,"utf-8");
//					BufferedWriter bufferedWriter = new BufferedWriter(osr);
					) {
	 
				for (String name : bboxList.keySet()) {
					
					String text = name;
					
					for(float val:(float[])bboxList.get(name)) {
						text += " " + val;
					}
					text += "\n";
//					System.out.println(text);
					fos.write(text.getBytes());
				}
	 
				fos.flush();
			} catch (Exception e) {
				e.printStackTrace();
			}
			
			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
//		String testPath = "I:\\000005.jpg";
//		
//		String outPath = "I:\\_000005.jpg";
//		
//		try {
//			YoloImageUtils.resize(testPath, outPath, 448, 448);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
//		
//		String testXML = "I:\\000005.xml";
//		
//		int[][] label = anno2bbox(testXML);
//		int[][] bbox = formatLabel(label);
//		System.out.println(JsonUtils.toJson(label));
//		System.out.println(JsonUtils.toJson(bbox));
		
		String rootPath = "H:\\voc\\train";
//		String filename = "000005.jpg";
		String imgDir = rootPath + "\\JPEGImages";
		String labelDir = rootPath + "\\Annotations";
		String imgOutDir = rootPath + "\\imgs";
		String bboxPath = rootPath + "\\labels\\bbox.txt";
		
//		formatData(filename, imgDir, labelDir, imgOutDir, true);
		image2Yolo(imgDir, labelDir, imgOutDir, bboxPath);
	}
	
}
