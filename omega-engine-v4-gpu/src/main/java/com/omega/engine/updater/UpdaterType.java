package com.omega.engine.updater;

/**
 * UpdaterType
 * @author Administrator
 *
 */
public enum UpdaterType {
	
	none("none"),
	momentum("momentum"),
	sgd("sgd"),
	adam("adam"),
	adamw("adamw");

	UpdaterType(String key){
		this.key = key;
	}
	
	private String key;

	public String getKey() {
		return key;
	}

}
