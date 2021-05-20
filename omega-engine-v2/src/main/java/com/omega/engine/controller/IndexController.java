package com.omega.engine.controller;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class IndexController {
	
	@RequestMapping("/")
	public String index(HttpServletRequest request,HttpServletResponse response) {
		
		return "index";
	}
	
	@RequestMapping("/AICar")
	public String aiCar(HttpServletRequest request,HttpServletResponse response) {
		
		return "2DCar/AICarDefautMap";
	}
	
}