package com.omega.example.transformer.utils.bpe;

import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import com.omega.common.task.ForkJobEngine;

/**
 * FileDataLoader
 * @author Administrator
 *
 */
public class EncodeEx extends RecursiveAction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6302699701667951010L;

	private int start = 0;
	
	private int end = 0;
	
	private List<String> txtList;
	
	private BPETokenizer3 bpe;
	
	private String[] idxList;
	
	private static EncodeEx job;
	
	public static EncodeEx getInstance(List<String> txtList,String[] idxList,BPETokenizer3 bpe,int start,int end) {
		if(job == null) {
			job = new EncodeEx(txtList, idxList, bpe, start, end);
		}else {
			if(txtList != job.getTxtList()){
				job.setTxtList(txtList);
			}
			job.setIdxList(idxList);
			job.setBpe(bpe);
			job.setStart(0);
			job.setEnd(end);
			job.reinitialize();
		}
		return job;
	}
	
	public EncodeEx(List<String> txtList,String[] idxList,BPETokenizer3 bpe,int start,int end) {
		this.setStart(start);
		this.setEnd(end);
		this.setIdxList(idxList);
		this.txtList = txtList;
		this.bpe = bpe;
	}
	
	@Override
	protected void compute() {
		// TODO Auto-generated method stub
		int length = getEnd() - getStart() + 1;
		
		if (length < 8 || length <= txtList.size() / 8) {
			
			load();

		} else {

			int mid = (getStart() + getEnd() + 1) >>> 1;
			EncodeEx left = new EncodeEx(txtList, idxList, bpe, getStart(), mid - 1);
			EncodeEx right = new EncodeEx(txtList, idxList, bpe, mid, getEnd());

			ForkJoinTask<Void> leftTask = left.fork();
			ForkJoinTask<Void> rightTask = right.fork();

			leftTask.join();
			rightTask.join();
			
		}
	}
	
	private void load() {
		
		for (int i = getStart(); i <= getEnd(); i++) {
			List<Integer> ids = bpe.encode(txtList.get(i));
			String txt = "";
			for(Integer id:ids) {
				txt += id + " ";
			}
			idxList[i] = txt;
//			System.out.println("encode["+i+"]finish.");
		}
		
	}
	
	public static void encode(List<String> txtList,String[] idxList,BPETokenizer3 bpe) {
//		System.out.println("encoding.");
		EncodeEx job = getInstance(txtList, idxList, bpe, 0, txtList.size() - 1);
		ForkJobEngine.run(job);
//		System.out.println("encode finish.");
	}
	
	public int getStart() {
		return start;
	}

	public void setStart(int start) {
		this.start = start;
	}

	public int getEnd() {
		return end;
	}

	public void setEnd(int end) {
		this.end = end;
	}

	public List<String> getTxtList() {
		return txtList;
	}

	public void setTxtList(List<String> txtList) {
		this.txtList = txtList;
	}

	public BPETokenizer3 getBpe() {
		return bpe;
	}

	public void setBpe(BPETokenizer3 bpe) {
		this.bpe = bpe;
	}

	public void setIdxList(String[] idxList) {
		this.idxList = idxList;
	}
	
}
