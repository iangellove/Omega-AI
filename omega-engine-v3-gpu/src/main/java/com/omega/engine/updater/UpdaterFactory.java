package com.omega.engine.updater;

import com.omega.engine.nn.network.Network;

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
	public static Updater create(UpdaterType type,Network net) {
		
		switch (type) {
		case momentum:
			return new Momentum();
		case adam:
			return new Adam(net);
		default:
			return null;
		}
		
	}
	
	
}
