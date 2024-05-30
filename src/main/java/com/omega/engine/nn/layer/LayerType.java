package com.omega.engine.nn.layer;

/**
 * layer type
 * @author Administrator
 * 
 */
public enum LayerType {
	
	full("full"),
	softmax("softmax"),
	conv("conv"),
	conv_transpose("conv_transpose"),
	cbl("cbl"),
	pooling("pooling"),
	input("input"),
	softmax_cross_entropy("softmax_cross_entropy"),
	sigmod("sigmod"),
	relu("relu"),
	gelu("gelu"),
	leakyRelu("leakyRelu"),
	tanh("tanh"),
	silu("silu"),
	bn("bn"),
	layer_norm("layer_norm"),
	instance_normal("instance_normal"),
	block("block"),
	shortcut("shortcut"),
	avgpooling("avgpooling"),
	upsample("upsample"),
	route("route"),
	yolo("yolo"),
	rnn("rnn"),
	mutli_head_attention("mutli_head_attention"),
	poswise_feed_forward("poswise_feed_forward"),
	mlp("mlp"),
	transformer_decoder("transformer_decoder"),
	transformer_block("transformer_block"),
	lstm("lstm"),
	embedding("embedding"),
	dropout("dropout");
	
	private String key;
	
	LayerType(String key){
		this.key = key;
	}

	public String getKey() {
		return key;
	}
	
	public static LayerType getEnumByKey(String key){
		for(LayerType type:LayerType.values()) {
			if(type.getKey().equals(key)) {
				return type;
			}
		}
		return null;
	}

}
