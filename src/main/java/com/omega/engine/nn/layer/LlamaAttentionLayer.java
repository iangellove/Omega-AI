package com.omega.engine.nn.layer;

import java.io.IOException;
import java.io.RandomAccessFile;

import com.omega.common.data.Tensor;

public abstract class LlamaAttentionLayer extends Layer{
	
	public abstract void forward(Tensor cos,Tensor sin,Tensor input);
	
	public abstract void back(Tensor cos,Tensor sin,Tensor delta);
	
	public abstract void saveModel(RandomAccessFile outputStream) throws IOException;
	
	public abstract void loadModel(RandomAccessFile inputStream) throws IOException;
	
}	
