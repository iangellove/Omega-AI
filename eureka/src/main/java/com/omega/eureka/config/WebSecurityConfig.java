//package com.omega.eureka.config;
//
//import org.springframework.beans.factory.annotation.Autowired;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.context.annotation.Configuration;
//import org.springframework.security.config.annotation.web.builders.HttpSecurity;
//import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
//import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
//
//import com.omega.eureka.handlers.FailHandler;
//import com.omega.eureka.handlers.SuccessHandler;
//
//
//@Configuration
//@EnableWebSecurity
//public class WebSecurityConfig extends WebSecurityConfigurerAdapter {
//	
//	@Value("${spring.security.user.name}")
//	private String username;
//	
//	@Value("${spring.security.user.password}")
//	private String password;
//	
//	@Autowired
//	private SuccessHandler successHandler;
//	
//	@Autowired
//	private FailHandler failHandler;
//	
////	@Override
////	protected void configure(AuthenticationManagerBuilder auth) throws Exception {
////	        auth.inMemoryAuthentication().passwordEncoder(new BCryptPasswordEncoder())
////	                .withUser(username).password(new BCryptPasswordEncoder().encode(password)).roles("ADMIN");
////	}
////	
////	@Override
////    protected void configure(HttpSecurity http) throws Exception {
//////        super.configure(http);//加这句是为了访问eureka控制台和/actuator时能做安全控制
////		http.formLogin().loginPage("/login/page").loginProcessingUrl("/login/login")
////		.usernameParameter("username")
////        .passwordParameter("password")
////        .successHandler(successHandler)
////        .failureHandler(failHandler)
////        .and()
////		.authorizeRequests()
////
////        // 只开启eureka注册账密校验
////        .antMatchers("/actuator/**","/login/**","/static/**").permitAll()
////        // 其他请求全放过
////        .anyRequest().authenticated()
////        .and().csrf().disable()//关闭csrf
////        // 开启基本账密校验
////        .httpBasic();
////    }
//	
//	 @Override
//	 protected void configure(HttpSecurity http) throws Exception {
//	      http.csrf().disable();
//	      super.configure(http);
//	 }
//	
//}
