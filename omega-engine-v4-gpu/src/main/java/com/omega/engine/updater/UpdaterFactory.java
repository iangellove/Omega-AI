package com.omega.engine.updater;

import java.util.Map;

/**
 * Updater Factory
 * @author Administrator
 * none
 * momentum
 * adam
 */
public class UpdaterFactory {
	
	/**
	 * create instance
	 * @param type
	 * @return
	 * none null
	 * momentum
	 * adam
	 */
	public static Updater create(UpdaterType type,Map<String,Float> params) {
		
		switch (type) {
		case momentum:
			return new Momentum();
		case sgd:
			return new SGDM();
		case adam:
			return new Adam();
		case adamw:
			return new AdamW(params);
		case RMSProp:
			return new RMSProp();
		default:
			return null;
		}
		
	}
	
	
}
