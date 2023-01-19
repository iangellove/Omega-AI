package com.omega.engine.gpu;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuModuleLoad;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

public class CUDAModules {
	
	public static Map<String,MyCUDAModule> modules = new HashMap<String,MyCUDAModule>();
	
	private static CUdevice device;
	
	private static CUcontext context;
	
	private static CUDAUtils instance;
	
	public static int maxThreads;
	
	public static int threadsPerDimension;
	
	public static Map<String,String> functions = new HashMap<String, String>(){
		/**
		 * 
		 */
		private static final long serialVersionUID = -7636602208380901817L;

		{
			put("col2im_gpu_kernelV2", "H://Col2imKernel.cu");
			put("im2col_gpu_kernelV2", "H://Im2colKernel.cu");
			put("pooling_diff", "H://PoolingKernel.cu");
			put("max_pooling", "H://PoolingKernel.cu");
			put("mean_pooling", "H://PoolingKernel.cu");
			put("mean_cov", "H://MathKernel.cu");
			put("fast_mean_kernel", "H://MathKernel.cu");
			put("var_cov", "H://MathKernel.cu");
			put("fast_variance_kernel", "H://MathKernel.cu");
			put("normalize_kernel", "H://BNKernel.cu");
			put("std_fn", "H://MathKernel.cu");
			put("mwa", "H://MathKernel.cu");
			put("culOutput_cov", "H://BNKernel.cu");
			put("computeDelta", "H://BNKernel.cu");
			put("computeDelta_full", "H://BNKernel.cu");
			put("meanDzSum", "H://BNKernel.cu");
			put("computeDiff", "H://BNKernel.cu");
			put("dgama_kernel", "H://BNKernel.cu");
			put("dbeta_kernel", "H://BNKernel.cu");
			put("dxhat_kernel2", "H://BNKernel.cu");
			put("full_mean_delta_kernel", "H://BNKernel.cu");
			put("full_var_delta_kernel", "H://BNKernel.cu");
			put("fast_variance_delta_kernel", "H://BNKernel.cu");
			put("dx_kernel", "H://BNKernel.cu");
			put("dx_kernel_full", "H://BNKernel.cu");
		}
		};
	
	
	public static CUfunction getFunctionByModule(String fileName,String functionName) {
		
		MyCUDAModule m = CUDAModules.getModule(fileName);
		
		if(m.getFunctions().containsKey(functionName)) {
			return m.getFunctions().get(functionName);
		}
		
		CUfunction function = new CUfunction();
        cuModuleGetFunction(function, m, functionName);
        m.getFunctions().put(functionName, function);
        
		return function;
	}
	
	public static MyCUDAModule getModule(String fileName) {
		
		// Create the PTX file by calling the NVCC
        try {
        	
			String ptxFileName = preparePtxFile(fileName);
			
			if(CUDAModules.modules.containsKey(ptxFileName)) {
				return CUDAModules.modules.get(ptxFileName);
			}
//	        
//	        long[] l = new long[1];
//	        int[] count = new int[1];
//	        JCudaDriver.cuDeviceTotalMem(l, device);
//	        JCudaDriver.cuDeviceGetCount(count);
//	        
//	        CUdevprop prop = new CUdevprop();
//	        JCudaDriver.cuDeviceGetProperties(prop, device);
//	        
//	        System.out.println(JsonUtils.toJson(l));
//	        System.out.println(JsonUtils.toJson(count));
//	        System.out.println(prop.toString());
	        
//        	JCudaDriver.setExceptionsEnabled(true);
//        	
//			// Initialize the driver and create a context for the first device.
//
//        	CUDAUtils instance = CUDAUtils.getInstance();
//        	
//        	instance.initCUDA();
//        	
//        	device = instance.getDevice(0);
//        	
//        	context = instance.getContext(device);

        	setContext(getContext());
        	
        	maxThreads = instance.getMaxThreads(device);
        	
            threadsPerDimension = (int) Math.sqrt(maxThreads);
			
	        // Load the ptx file.
	        MyCUDAModule module = new MyCUDAModule();
	        cuModuleLoad(module, ptxFileName);
	        
	        CUDAModules.modules.put(ptxFileName, module);
			
			return module;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
        return null;
	}
	
    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException{
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }
        System.out.println(ptxFileName);
        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
            new String(toByteArray(process.getErrorStream()));
        String outputMessage =
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

	public static CUcontext getContext() {
		
		if(context == null) {
			
			JCudaDriver.setExceptionsEnabled(true);
        	
			// Initialize the driver and create a context for the first device.

        	instance = CUDAUtils.getInstance();
        	
        	instance.initCUDA();
        	
        	device = instance.getDevice(0);
        	
        	context = instance.getContext(device);
			
        	System.out.println("CUDA context init finish.");
        	
		}
		
		return context;
	}
	
	public static void initContext() {
		getContext();
	}

	public static void setContext(CUcontext context) {
		CUDAModules.context = context;
	}
	
	public static void initCUDAFunctions() {
		for(String key:functions.keySet()) {
			CUDAModules.getFunctionByModule(functions.get(key), key);
		}
		System.out.println("CUDA functions init finish.");
	}
	
}