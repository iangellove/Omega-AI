package com.omega.engine.database;

import java.util.HashMap;
import java.util.Map;

import com.omega.engine.nn.network.Network;

public class NetworksDataBase {
	
	private Map<String,Network> networks = new HashMap<String, Network>();

	public Map<String,Network> getNetworks() {
		return networks;
	}

	public void setNetworks(Map<String,Network> networks) {
		this.networks = networks;
	}

}
