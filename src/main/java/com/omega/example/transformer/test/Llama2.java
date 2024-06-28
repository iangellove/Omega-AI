package com.omega.example.transformer.test;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

import com.omega.common.utils.JsonUtils;

public class Llama2 {
	
	 // 0: OFF, 1: ERROR, 2: INFO, 3: DEBUG
    static final int LOG_LEVEL = Integer.parseInt(System.getProperty("log.level", "2"));

    static final DateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS");
	
	class Tokenizer {

	    String[] vocab;
	    float[] vocab_scores;
	    Map<String, Integer> sorted_vocab;
	    int vocab_size;
	    int max_token_length;
	    String[] byte_pieces = new String[256]; // stores all single-byte strings
	}
	
	static void build_tokenizer(Tokenizer t, String tokenizer_path, int vocab_size) {
        t.vocab_size = vocab_size;
        // malloc space to hold the scores and the strings
        t.vocab = new String[vocab_size];
        t.vocab_scores = new float[vocab_size];

        for (int i = 0; i < 256; i++) {
            t.byte_pieces[i] = String.valueOf((char) i);
        }
        try (FileChannel file = new FileInputStream(tokenizer_path).getChannel()) {
            ByteBuffer buffer = file.map(FileChannel.MapMode.READ_ONLY, 0, file.size());
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            t.max_token_length = buffer.getInt();
            System.out.println("max_token_length:"+t.max_token_length);
            int len;
            for (int i = 0; i < vocab_size; i++) {
            	System.out.println(i);
                t.vocab_scores[i] = buffer.getFloat();
                System.out.println(t.vocab_scores[i]);
                len = buffer.getInt();
                byte[] vocabBytes = new byte[len];
                buffer.get(vocabBytes);
                t.vocab[i] = new String(vocabBytes);
                System.out.println(t.vocab[i]);
            }
        } catch (IOException e) {
            throw new RuntimeException("couldn't load " + tokenizer_path, e);
        }

        //t.sorted_vocab = null; // initialized lazily
        t.sorted_vocab = new HashMap<>(t.vocab_size * 4 / 3);
        for (int i = 0; i < vocab_size; i++) {
            t.sorted_vocab.put(t.vocab[i], i);
        }
        System.out.println(JsonUtils.toJson(t));
        System.out.println("Build tokenizer successfully, vocab_size=" + vocab_size
                + ", max_token_length=" + t.max_token_length);
//        logDebug();
    }

	static void logDebug(String s) {
        if (LOG_LEVEL > 2) {
            System.out.println(DATE_FORMAT.format(new Date()) + " [DEBUG] " + s);
        }
    }
	
    static int str_lookup(String str, Map<String, Integer> sorted_vocab) {
        // efficiently find the perfect match for str in vocab, return its index or -1 if not found
        return sorted_vocab.getOrDefault(str, -1);
    }
	
    static String decode(Tokenizer t, int prev_token, int token) {
        String piece = t.vocab[token];
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if (prev_token == 1 && piece.charAt(0) == ' ') {
            piece = piece.substring(1);
        }

        if (piece.length() == 6
                && piece.charAt(0) == '<'
                && piece.charAt(1) == '0'
                && piece.charAt(2) == 'x'
                && piece.charAt(5) == '>') {
            int byte_val = Integer.parseInt(piece.substring(3, 5), 16);
            piece = t.byte_pieces[byte_val];
        }
        return piece;
    }
    
	static int encode(Tokenizer t, String text, boolean bos, boolean eos, int[] tokens, int num_prompt_tokens) {
	        // encode the string text (input) into an upper-bound preallocated tokens[] array
	        // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
	        if (text == null) {
	            System.out.println("cannot encode NULL text");
	            System.exit(1);
	        }

	        // start at 0 tokens
	        int n_tokens = 0;

	        // add optional BOS (=1) token, if desired
	        if (bos) {
	            tokens[n_tokens++] = 1;
	        }

	        // add_dummy_prefix is true by default
	        // so prepend a dummy prefix token to the input string, but only if text != ""
	        // TODO: pretty sure this isn't correct in the general case but I don't have the
	        // energy to read more of the sentencepiece code to figure out what it's doing
	        if (!text.isEmpty()) {
	            int dummy_prefix = str_lookup(" ", t.sorted_vocab);
	            tokens[n_tokens++] = dummy_prefix;
	        }

	        // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
	        // Code point ↔ UTF-8 conversion
	        // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
	        // U+0000	U+007F	    0xxxxxxx
	        // U+0080	U+07FF	    110xxxxx	10xxxxxx
	        // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
	        // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

	        // process the raw (UTF-8) byte sequence of the input string
	        for (int i = 0, cpi; i < text.length(); i += Character.charCount(cpi)) {
	            cpi = text.codePointAt(i);

	            String singleCodepoint =  Character.toString(Character.highSurrogate(cpi));

	            // ok c+1 is not a continuation byte, so we've read in a full codepoint
	            int id = str_lookup(singleCodepoint, t.sorted_vocab);

	            if (id != -1) {
	                // we found this codepoint in vocab, add it as a token
	                tokens[n_tokens++] = id;
	            } else {
	                // byte_fallback encoding: just encode each byte as a token
	                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
	                // so the individual bytes only start at index 3
	                for (byte b : singleCodepoint.getBytes(StandardCharsets.UTF_8)) {
	                    tokens[n_tokens++] = Byte.toUnsignedInt(b) + 3;
	                }
	            }
	        }

	        // merge the best consecutive pair each iteration, according the scores in vocab_scores
	        while (true) {
	            float best_score = -1e10f;
	            int best_id = -1;
	            int best_idx = -1;

	            for (int i = 0; i < n_tokens - 1; i++) {
	                // check if we can merge the pair (tokens[i], tokens[i+1])
	                String str_buf = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
	                int id = str_lookup(str_buf, t.sorted_vocab);
	                if (id != -1 && t.vocab_scores[id] > best_score) {
	                    // this merge pair exists in vocab! record its score and position
	                    best_score = t.vocab_scores[id];
	                    best_id = id;
	                    best_idx = i;
	                }
	            }

	            if (best_idx == -1) {
	                break; // we couldn't find any more pairs to merge, so we're done
	            }

	            // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
	            tokens[best_idx] = best_id;
	            // delete token at position best_idx+1, shift the entire sequence back 1
	            for (int i = best_idx + 1; i < n_tokens - 1; i++) {
	                tokens[i] = tokens[i + 1];
	            }
	            n_tokens--; // token length decreased
	        }

	        // add optional EOS (=2) token, if desired
	        if (eos) {
	            tokens[n_tokens++] = 2;
	        }

	        return n_tokens;
	}
	
	public static void main(String[] args) {
		
		String tokenizer_path = "H:\\transformer_dataset\\tokenizer.bin";
		
		Llama2 l = new Llama2();
		
		Tokenizer t = l.new Tokenizer();
		
		build_tokenizer(t, tokenizer_path, 64793);
		
		
	}
	
}
