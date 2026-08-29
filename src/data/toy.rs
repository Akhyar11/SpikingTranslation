use rand::Rng;

/// Struktur Toy Dataset sederhana.
/// Digunakan untuk uji coba flow (aliran data) dari SNN.
/// Input: Sequence token acak, Target: Sequence input yang digeser 1 step (mirip task bahasa).
pub struct ToyDataset {
    pub vocab_size: usize,
    pub seq_length: usize,
    pub num_samples: usize,
}

impl ToyDataset {
    pub fn new(vocab_size: usize, seq_length: usize, num_samples: usize) -> Self {
        ToyDataset {
            vocab_size,
            seq_length,
            num_samples,
        }
    }

    /// Menghasilkan pasangan data source dan target
    pub fn generate_batch(&self, batch_size: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let mut rng = rand::thread_rng();
        let mut sources = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let mut src = Vec::with_capacity(self.seq_length);
            let mut tgt = Vec::with_capacity(self.seq_length);

            for _i in 0..self.seq_length {
                // Token dimulai dari indeks 1 (0 bisa digunakan untuk padding/spesial)
                let token = rng.gen_range(1..self.vocab_size);
                src.push(token);
                // Target bisa dibuat sederhana, misalnya token input + 1 (looping)
                let tgt_token = if token + 1 >= self.vocab_size { 1 } else { token + 1 };
                tgt.push(tgt_token);
            }
            
            sources.push(src);
            targets.push(tgt);
        }

        (sources, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_dataset_generation() {
        let dataset = ToyDataset::new(100, 5, 10); // vocab 100, seq_len 5, batch 10
        let (src, tgt) = dataset.generate_batch(10);
        
        assert_eq!(src.len(), 10);
        assert_eq!(tgt.len(), 10);
        assert_eq!(src[0].len(), 5);
        assert_eq!(tgt[0].len(), 5);

        // Verifikasi bahwa target adalah pergeseran dari source (src + 1)
        for b in 0..10 {
            for i in 0..5 {
                let expected_tgt = if src[b][i] + 1 >= 100 { 1 } else { src[b][i] + 1 };
                assert_eq!(tgt[b][i], expected_tgt);
            }
        }
    }
}
