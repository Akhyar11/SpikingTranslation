use std::fs::File;
use std::io::{BufRead, BufReader};
use tokenizers::Tokenizer;

pub struct StreamingCorpus {
    pub tokenizer: Tokenizer,
    src_path: String,
    tgt_path: String,
}

impl StreamingCorpus {
    pub fn new(src_path: &str, tgt_path: &str) -> Self {
        let tokenizer = Tokenizer::from_file("bpe_tokenizer.json").unwrap();
        StreamingCorpus {
            tokenizer,
            src_path: src_path.to_string(),
            tgt_path: tgt_path.to_string(),
        }
    }
    
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
    
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        encoding.get_ids().iter().map(|&x| x as usize).collect()
    }
    
    pub fn decode(&self, ids: &[usize]) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        self.tokenizer.decode(&ids_u32, true).unwrap_or_default()
    }
}
