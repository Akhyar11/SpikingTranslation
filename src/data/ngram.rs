use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub struct SparseNGramMemory {
    /// Bigram memory: given token i-1, what is the frequency of token i
    /// Mapping: prev_token_id -> HashMap<curr_token_id, count>
    pub bigram_counts: HashMap<usize, HashMap<usize, usize>>,
}

impl SparseNGramMemory {
    pub fn new() -> Self {
        SparseNGramMemory {
            bigram_counts: HashMap::new(),
        }
    }

    /// Membangun statis memory Bigram berdasarkan BPE tokenized sequences.
    /// Memori ini sangat sparse, dan hanya mencatat kata-kata yang benar-benar ada di corpus target.
    pub fn build_from_corpus(&mut self, corpus: &crate::data::corpus::StreamingCorpus, max_lines: usize) {
        println!("Membangun Sparse N-Gram Memory dari corpus target (Max {} baris)...", max_lines);
        let tgt_file = File::open("dataset/OpenSubtitles.en-id.id").expect("Gagal membuka file target!");
        let reader = BufReader::new(tgt_file);

        let mut count = 0;
        let eos_id = corpus.get_eos_id();

        for line in reader.lines().take(max_lines) {
            if let Ok(text) = line {
                let mut tokens = corpus.encode(&text);
                tokens.push(eos_id);

                if tokens.len() > 1 {
                    for i in 1..tokens.len() {
                        let prev = tokens[i - 1];
                        let curr = tokens[i];

                        let entry = self.bigram_counts.entry(prev).or_insert_with(HashMap::new);
                        *entry.entry(curr).or_insert(0) += 1;
                    }
                }
                count += 1;
            }
        }
        println!("Sparse N-Gram Memory selesai dibangun dari {} kalimat.", count);
        println!("Total prefix Bigram keys: {}", self.bigram_counts.len());
    }

    /// Mendapatkan candidate tokens berdasarkan prefix sebelumnya (Bigram)
    /// Mengembalikan token dan probabilitas kasarnya (count).
    pub fn get_candidates(&self, prev_token: usize) -> Vec<(usize, f32)> {
        if let Some(next_counts) = self.bigram_counts.get(&prev_token) {
            let total: usize = next_counts.values().sum();
            let mut candidates: Vec<(usize, f32)> = next_counts.iter()
                .map(|(&tok, &cnt)| (tok, cnt as f32 / total as f32))
                .collect();
            // Urutkan berdasarkan probabilitas tertinggi
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates
        } else {
            Vec::new()
        }
    }
}
