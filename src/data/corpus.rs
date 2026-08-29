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
        let tokenizer = Tokenizer::from_file("bpe_tokenizer.json")
            .expect("Gagal memuat bpe_tokenizer.json. Harap latih tokenizer terlebih dahulu!");
            
        StreamingCorpus {
            tokenizer,
            src_path: src_path.to_string(),
            tgt_path: tgt_path.to_string(),
        }
    }

    /// Mendapatkan ukuran vocab yang sudah dilatih BPE
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Kompatibilitas untuk kode lama (biar nggak banyak error di tempat lain)
    pub fn get_vocab_len(&self) -> usize {
        self.vocab_size()
    }
    
    pub fn get_pad_id(&self) -> usize {
        self.tokenizer.token_to_id("<PAD>").unwrap_or(0) as usize
    }

    pub fn get_eos_id(&self) -> usize {
        self.tokenizer.token_to_id("<EOS>").unwrap_or(2) as usize
    }

    /// Encoding string ke vector of indices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        // lower casing manually as our simple tokenizer python script might not do it implicitly
        let text = text.to_lowercase();
        let text = text.replace(&['.', ',', '!', '?', '"', '\''][..], "");
        let encoding = self.tokenizer.encode(text, true).unwrap();
        encoding.get_ids().iter().map(|&x| x as usize).collect()
    }
    
    pub fn decode(&self, ids: &[usize]) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        self.tokenizer.decode(&ids_u32, true).unwrap_or_default()
    }

    /// Pass 1: Membangun kosata secara streaming dihapus karena BPE Tokenizer sudah dilatih offline.
    pub fn build_vocab(&mut self, _max_lines: usize) {
        println!("Vocab menggunakan BPE Tokenizer offline.");
        println!("Vocab Size: {}", self.vocab_size());
    }

    /// Mendapatkan iterator streaming yang mengembalikan sekuens indeks per *batch*
    pub fn stream_batches(&self, batch_size: usize, limit: usize) -> CorpusBatchIterator<'_> {
        let src_file = File::open(&self.src_path).expect("Gagal membuka file sumber!");
        let tgt_file = File::open(&self.tgt_path).expect("Gagal membuka file target!");
        
        CorpusBatchIterator {
            src_reader: BufReader::new(src_file),
            tgt_reader: BufReader::new(tgt_file),
            corpus: self,
            batch_size,
            lines_read: 0,
            limit,
        }
    }
}

pub struct CorpusBatchIterator<'a> {
    src_reader: BufReader<File>,
    tgt_reader: BufReader<File>,
    corpus: &'a StreamingCorpus,
    batch_size: usize,
    lines_read: usize,
    limit: usize,
}

impl<'a> Iterator for CorpusBatchIterator<'a> {
    type Item = (Vec<Vec<usize>>, Vec<Vec<usize>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.lines_read >= self.limit {
            return None;
        }

        let mut sources = Vec::with_capacity(self.batch_size);
        let mut targets = Vec::with_capacity(self.batch_size);
        
        let eos_id = self.corpus.get_eos_id();

        for _ in 0..self.batch_size {
            if self.lines_read >= self.limit { break; }

            let mut src_line = String::new();
            let mut tgt_line = String::new();

            let src_bytes = self.src_reader.read_line(&mut src_line).unwrap_or(0);
            let tgt_bytes = self.tgt_reader.read_line(&mut tgt_line).unwrap_or(0);

            if src_bytes == 0 || tgt_bytes == 0 {
                break; // EOF
            }

            let mut src_indices = self.corpus.encode(&src_line);
            src_indices.push(eos_id);

            let mut tgt_indices = self.corpus.encode(&tgt_line);
            tgt_indices.push(eos_id);

            sources.push(src_indices);
            targets.push(tgt_indices);
            
            self.lines_read += 1;
        }

        if sources.is_empty() {
            None
        } else {
            Some((sources, targets))
        }
    }
}
