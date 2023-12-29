package com.omega.eureka.controller;

import javax.servlet.http.HttpServletRequest;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;

@Controller
@RequestMapping(value = "/login", method = { RequestMethod.POST, RequestMethod.GET })
public class LoginController {
	
	@RequestMapping("/page")
	public String page(HttpServletRequest request){
		
		return "/login.html";
	} 
	
//	@ResponseBody
//	@RequestMapping("/login")
//	public Map<String,Object> login(HttpServletRequest request){
//		
//		Map<String,Object> result = new HashMap<String, Object>();
//		
//		try {
//			
//			String username = request.getParameter("username");
//			String password = request.getParameter("password");
//			
//			System.out.println("username:"+username+",password:"+password);
//			
//			result.put("success", true);
//			result.put("code", 200);
//			result.put("msg", "登录成功");
//			return result;
//		} catch (Exception e) {
//			// TODO: handle exception
//			e.printStackTrace();
//			result.put("success", false);
//			result.put("code", 400);
//			result.put("msg", "查询失败");
//		}
//		
//		return result;
//	} 
	
}
