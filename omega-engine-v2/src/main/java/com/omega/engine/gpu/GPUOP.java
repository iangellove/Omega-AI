package com.omega.engine.gpu;

import static jcuda.jcublas.JCublas2.cublasCreate;
import static jcuda.jcublas.JCublas2.cublasDestroy;
import static jcuda.jcublas.JCublas2.cublasGetVector;
import static jcuda.jcublas.JCublas2.cublasSetVector;
import static jcuda.jcublas.JCublas2.cublasSgemm;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
//import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;

import java.util.Locale;
import java.util.Vector;

import com.omega.common.task.Task;
import com.omega.common.task.TaskEngine;
import com.omega.common.utils.JsonUtils;
import com.omega.common.utils.MatrixOperation;
import com.omega.common.utils.MatrixUtils;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;

public class GPUOP {
    
	private static GPUOP o;
	
	private cublasHandle handle;
	
	public GPUOP() {
		// TODO Auto-generated constructor stub
		this.handle = new cublasHandle();
	}
	
	/**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    public void multiplyFloat(int m,int n,int k, float A[],
        float B[], float C[])
    {
    	cublasCreate(handle);
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, m * n * Sizeof.FLOAT);
        cudaMalloc(dB, n * k * Sizeof.FLOAT);
        cudaMalloc(dC, m * k * Sizeof.FLOAT);
        cublasSetVector(m * n, Sizeof.FLOAT, Pointer.to(A), 1, dA, 1);
        cublasSetVector(n * k, Sizeof.FLOAT, Pointer.to(B), 1, dB, 1);
        cublasSetVector(m * k, Sizeof.FLOAT, Pointer.to(C), 1, dC, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        
        long start = System.nanoTime();
        
        int status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, 
            dB, k, dA, n, zero, dC, k);

        cudaDeviceSynchronize();
        
        long end = System.nanoTime();
        
        System.out.println((end - start) / 1e6+"ms");
        
        cublasGetVector(m * k, Sizeof.FLOAT, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
       
        // Clean up
        cublasDestroy(handle);
    }
    
	/**
     * Multiplies the matrices A and B and writes the result into C.
     * 
     * @param size The size of the matrices
     * @param A Matrix A
     * @param B Matrix B
     * @param C Matrix C
     */
    public void multiplyDouble( int m,int n,int k, double A[],
			double B[], double C[])
    {
    	cublasCreate(handle);
        Pointer dA = new Pointer();
        Pointer dB = new Pointer();
        Pointer dC = new Pointer();

        cudaMalloc(dA, m * n * Sizeof.DOUBLE);
        cudaMalloc(dB, n * k * Sizeof.DOUBLE);
        cudaMalloc(dC, m * k * Sizeof.DOUBLE);
        cublasSetVector(m * n, Sizeof.DOUBLE, Pointer.to(A), 1, dA, 1);
        cublasSetVector(n * k, Sizeof.DOUBLE, Pointer.to(B), 1, dB, 1);
        cublasSetVector(m * k, Sizeof.DOUBLE, Pointer.to(C), 1, dC, 1);

        Pointer zero = Pointer.to(new float[]{ 0.0f });
        Pointer one = Pointer.to(new float[]{ 1.0f });
        int status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, m, n, one, 
            dB, k, dA, n, zero, dC, k);

        cudaDeviceSynchronize();
        
        cublasGetVector(m * k, Sizeof.DOUBLE, dC, 1, Pointer.to(C), 1);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        // Clean up
        cublasDestroy(handle);
    }
    
    public static GPUOP getInstance() {
    	
    	if(GPUOP.o == null) {
    		GPUOP.o = new GPUOP();
    	}
    	
    	return GPUOP.o;
    }
    
    public static void main(String[] args){
    	
    	//1024,576,64
    	//256,576,128
    	//256,1152,128
    	//16,4608,512
        // 1024 * 27
        
        
//        System.out.println("a:");
//        System.out.println(GPUOP.toString2D(a, n));
//        System.out.println("b:");
//        System.out.println(GPUOP.toString2D(b, k));
//        System.out.println("c:");
//        System.out.println(GPUOP.toString2D(c, k));
        

		int m = 256;
        int n = 1152;
        int k = 128;
        float[] a = MatrixUtils.val(m * n, 1.0f);
        
        // 24 * 64
        float[] b = MatrixUtils.val(n * k, 2.0f);
        
        float[] c = new float [m * k];
        
        GPUOP.getInstance().multiplyFloat(m, n, k, a, b, c);
        
        
    }
    
    public static String toString2D(float[] a,int columns){
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
    public static String toString2D(double[] a,int columns){
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++)
        {
            if ((i > 0) && (i % columns == 0))
            {
                sb.append("\n");
            }
            sb.append(String.format(Locale.ENGLISH, "%7.4f ", a[i]));
        }
        return sb.toString();
    }
    
}
