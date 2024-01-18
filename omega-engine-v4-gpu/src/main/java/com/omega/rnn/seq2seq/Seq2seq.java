package com.omega.rnn.seq2seq;

import java.util.Scanner;

import com.omega.engine.gpu.CUDAMemoryManager;
import com.omega.engine.gpu.CUDAModules;
import com.omega.engine.loss.LossType;
import com.omega.engine.nn.network.Seq2Seq;
import com.omega.engine.optimizer.EDOptimizer;
import com.omega.engine.optimizer.lr.LearnRateUpdate;
import com.omega.engine.updater.UpdaterType;
import com.omega.rnn.data.IndexDataLoader;

public class Seq2seq {
	
	public void seq2seq() {
		
		try {
			
			int batchSize = 128;
			
			int en_em = 64;
			
			int de_em = 128;
			
			int en_hidden = 256;
			
			int de_hidden = 256;
			
			String trainPath = "H:\\rnn_dataset\\translate1000.csv";

			IndexDataLoader trainData = new IndexDataLoader(trainPath, batchSize);
			
			Seq2Seq network = new Seq2Seq(LossType.softmax_with_cross_entropy, UpdaterType.adamw,
					trainData.max_en, trainData.max_ch - 1, en_em, en_hidden, trainData.en_characters, de_em, de_hidden, trainData.ch_characters);
			
			network.CUDNN = true;
			
			network.learnRate = 0.01f;
			
			EDOptimizer optimizer = new EDOptimizer(network, batchSize, 100, 0.001f, LearnRateUpdate.SMART_HALF, false);
			optimizer.lr_step = new int[] {100,200};
			optimizer.trainRNN(trainData);

			Scanner scanner = new Scanner(System.in);
			
			while (true) {
				
				System.out.println("请输入英文:");
				String input_txt = scanner.nextLine();
				if(input_txt.equals("exit")){
					break;
				}
				input_txt = input_txt.toLowerCase();
				System.out.println(input_txt);
				optimizer.predict(trainData, input_txt);
				
			}
			
			scanner.close();
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}
		
	}
	
	public static void main(String[] args) {
		
		try {

			CUDAModules.initContext();
			
			Seq2seq t = new Seq2seq();

			t.seq2seq();

			
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		} finally {
			// TODO: handle finally clause
			CUDAMemoryManager.free();
		}
		
	}
	
}
