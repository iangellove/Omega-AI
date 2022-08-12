//package com.omega.eureka.handlers;
//
//import java.io.IOException;
//
//import javax.servlet.ServletException;
//import javax.servlet.http.HttpServletRequest;
//import javax.servlet.http.HttpServletResponse;
//
//import org.springframework.security.core.AuthenticationException;
//import org.springframework.security.web.authentication.AuthenticationFailureHandler;
//import org.springframework.stereotype.Component;
//
//@Component
//public class FailHandler implements AuthenticationFailureHandler{
//
//	@Override
//	public void onAuthenticationFailure(HttpServletRequest request, HttpServletResponse response,
//			AuthenticationException exception) throws IOException, ServletException {
//		// TODO Auto-generated method stub
//		System.out.println("login fail.");
//		response.sendRedirect("/login/page");
//	}
//
//}
