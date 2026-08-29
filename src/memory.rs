use std::collections::{HashMap, HashSet};

/// Struktur memori statis untuk N-Gram Lexical Retrieval.
/// Memetakan N-gram dari bahasa sumber ke kumpulan kandidat token pada bahasa target.
pub struct NGramMemory {
    /// Kunci: tuple token source (contoh: tupel ukuran 1 s/d max_n)
    /// Nilai: set token target yang menjadi kandidat terjemahannya
    pub store: HashMap<Vec<usize>, HashSet<usize>>,
    pub max_n: usize,
}

impl NGramMemory {
    pub fn new(max_n: usize) -> Self {
        Self {
            store: HashMap::new(),
            max_n,
        }
    }

    /// Membangun kamus n-gram dari pasangan kalimat source dan target.
    /// Ini mengimplementasikan "Fase 3.1".
    pub fn build_from_corpus(&mut self, corpus: &[(Vec<usize>, Vec<usize>)]) {
        for (source_sentence, target_sentence) in corpus {
            let target_set: HashSet<usize> = target_sentence.iter().copied().collect();
            
            // Ekstrak semua n-gram dari source sentence (n = 1 s/d max_n)
            for n in 1..=self.max_n {
                if source_sentence.len() >= n {
                    for window in source_sentence.windows(n) {
                        let ngram = window.to_vec();
                        self.store
                            .entry(ngram)
                            .or_insert_with(HashSet::new)
                            .extend(&target_set);
                    }
                }
            }
        }
    }

    /// Mencari (lookup) kandidat token target berdasarkan N-gram source statis tunggal.
    /// Ini mengimplementasikan "Fase 3.2".
    pub fn lookup(&self, source_ngram: &[usize]) -> HashSet<usize> {
        self.store.get(source_ngram).cloned().unwrap_or_default()
    }
    
    /// Mencari (lookup) gabungan semua kandidat token target 
    /// berdasarkan seluruh rentetan N-gram yang ada di sebuah kalimat source.
    pub fn get_candidates_for_sentence(&self, source_sentence: &[usize]) -> HashSet<usize> {
        let mut candidates = HashSet::new();
        for n in 1..=self.max_n {
            if source_sentence.len() >= n {
                for window in source_sentence.windows(n) {
                    if let Some(set) = self.store.get(window) {
                        candidates.extend(set);
                    }
                }
            }
        }
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ngram_memory_build_and_lookup() {
        let corpus = vec![
            (vec![1, 2, 3], vec![10, 20]),
            (vec![2, 3, 4], vec![20, 30]),
        ];

        let mut mem = NGramMemory::new(2);
        mem.build_from_corpus(&corpus);

        // N-gram [2, 3] muncul di kedua kalimat, jadi harus memetakan ke 10, 20, 30
        let lookup_23 = mem.lookup(&[2, 3]);
        assert!(lookup_23.contains(&10));
        assert!(lookup_23.contains(&20));
        assert!(lookup_23.contains(&30));

        // N-gram [1, 2] hanya muncul di kalimat pertama
        let lookup_12 = mem.lookup(&[1, 2]);
        assert!(lookup_12.contains(&10));
        assert!(lookup_12.contains(&20));
        assert!(!lookup_12.contains(&30));

        // Tes agregasi kalimat
        let sentence_candidates = mem.get_candidates_for_sentence(&[1, 2, 99]); // 99 tidak ada, tapi unigram [2] membawa 30
        assert!(sentence_candidates.contains(&10));
        assert!(sentence_candidates.contains(&20));
        assert!(sentence_candidates.contains(&30));
    }

    #[test]
    fn test_sparse_representation() {
        let corpus = vec![(vec![1, 2], vec![10, 20])];
        let mut mem = NGramMemory::new(2);
        mem.build_from_corpus(&corpus);

        let candidates = mem.get_candidates_for_sentence(&[1, 2]);
        
        // Memastikan tipe datanya adalah sekumpulan index spesifik (sparse)
        // Bukan vektor dense berukuran ukuran_vocabulary yang diisi 0 dan 1
        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&10));
        assert!(candidates.contains(&20));
        
        // Dalam implementasi SNN, HashSet ini mewakili M_v (kolom indeks neuron yang menyala)
    }
}
