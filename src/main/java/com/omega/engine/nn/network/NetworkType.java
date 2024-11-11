package com.omega.engine.nn.network;

/**
 * 
 * @author Administrator
 *
 */
public enum NetworkType {
	
	BP("BP"),
	CNN("CNN"),
	ANN("ANN"),
	RNN("RNN"),
	SEQ2SEQ_RNN("SEQ2SEQ_RNN"),
	SEQ2SEQ("SEQ2SEQ"),
	TTANSFORMER("TTANSFORMER"),
	GPT("GPT"),
	LLAMA("LLAMA"),
	LLAVA("LLAVA"),
	VAE("VAE"),
	VQVAE("VQVAE"),
	CLIP_VISION("CLIP_VISION"),
	UNET("UNET"),
	DUFFSION_UNET("DUFFSION_UNET"),
	YOLO("YOLO");
	
	NetworkType(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}

}
