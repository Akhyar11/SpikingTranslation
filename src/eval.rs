use std::collections::HashMap;
use crate::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};
use crate::data::ngram::SparseNGramMemory;
use ndarray::Array1;

pub fn infer(
    encoder: &SpikingEncoder,
    stcm: &STCM,
    decoder: &SpikingDecoder,
    src_seq: &[usize],
    max_len: usize,
    k: usize,
    ngram_memory: Option<&SparseNGramMemory>,
) -> Vec<usize> {
    // 1. Encode Source
    let mut s_e_prev = vec![Array1::zeros(encoder.w_e.raw_dim()[0]); k];
    let mut u_e_prev = Array1::zeros(encoder.w_e.raw_dim()[0]);
    let mut s_c_prev = vec![Array1::zeros(stcm.w_cc.raw_dim()[0]); k];
    let mut u_c_prev = Array1::zeros(stcm.w_cc.raw_dim()[0]);
    
    // Tidak ada parameter last_s_c di forward_decoder_token,
    // kita hanya melacak u dan s.
    
    for &token in src_seq {
        let mut s_x = vec![Array1::zeros(encoder.w_e.raw_dim()[1]); k];
        if token < s_x[0].len() {
            for tau in 0..k { s_x[tau][token] = 1.0; }
        }
        
        let (u_e, s_e) = encoder.forward_token(&s_x, &s_e_prev, &u_e_prev);
        let (u_c, s_c) = stcm.forward_source_token(&s_e, &s_c_prev, &u_c_prev);
        
        s_e_prev = s_e;
        u_e_prev = u_e.last().unwrap().clone();
        s_c_prev = s_c;
        u_c_prev = u_c.last().unwrap().clone();
    }
    
    // 2. Decode Autoregressively
    let mut s_ctx_prev = s_c_prev;
    let mut u_ctx_prev = u_c_prev;
    let mut s_d_prev = vec![Array1::zeros(decoder.w_r.raw_dim()[0]); k];
    let mut u_d_prev = Array1::zeros(decoder.w_r.raw_dim()[0]);
    
    let mut result = Vec::new();
    let mut current_token = 2; // Asumsi <EOS> atau <BOS>
    
    for _ in 0..max_len {
        let mut s_y_prev = vec![Array1::zeros(decoder.w_y.raw_dim()[1]); k];
        if current_token < s_y_prev[0].len() {
            for tau in 0..k { s_y_prev[tau][current_token] = 1.0; }
        }
        
        // forward_decoder_token hanya menerima 3 argumen: s_d_prev_t, s_ctx_prev_t, u_prev_k
        let (u_ctx, s_ctx) = stcm.forward_decoder_token(&s_d_prev, &s_ctx_prev, &u_ctx_prev);
        let (u_d, s_d) = decoder.forward_token(&s_y_prev, &s_ctx, &s_d_prev, &u_d_prev);
        
        // Prediksi token menggunakan akumulasi spike membran (s_d) dan pemetaan SDR
        let vocab_tgt = decoder.w_y.raw_dim()[1];
        let d_d = decoder.w_r.raw_dim()[0];

        let mut neuron_s_sums = vec![0.0; d_d];
        for tau in 0..k {
            for (i, val) in s_d[tau].iter().enumerate() {
                neuron_s_sums[i] += val;
            }
        }
        
        let mut token_scores = vec![0.0; vocab_tgt];
        for v in 0..vocab_tgt {
            let m_v = crate::snn::architecture::sdr_token_map(v, d_d, 3);
            for &neuron_id in &m_v {
                token_scores[v] += neuron_s_sums[neuron_id];
            }
        }
        
        // --- N-Gram Memory Integration ---
        if let Some(mem) = ngram_memory {
            let candidates = mem.get_candidates(current_token);
            for (tok, prob) in candidates {
                if tok < token_scores.len() {
                    // Boost skor berdasar probabilitas Bigram Memory
                    token_scores[tok] += prob * 2.0;
                }
            }
        }
        
        let mut best_token = 1; // <UNK>
        let mut max_score = f32::MIN;
        for (i, &s) in token_scores.iter().enumerate() {
            // Abaikan token PAD (0) dan UNK (1) dari prediksi langsung jika memungkinkan
            if i < 2 { continue; } 
            
            if s > max_score {
                max_score = s;
                best_token = i;
            }
        }
        
        result.push(best_token);
        current_token = best_token;
        
        s_ctx_prev = s_ctx;
        u_ctx_prev = u_ctx.last().unwrap().clone();
        s_d_prev = s_d;
        u_d_prev = u_d.last().unwrap().clone();
        
        if current_token == 2 { // <EOS>
            break;
        }
    }
    
    result
}

/// Menghitung BLEU-4 Score untuk kumpulan hasil prediksi (hypotheses) terhadap target aslinya (references)
pub fn calculate_bleu(references: &[Vec<usize>], hypotheses: &[Vec<usize>]) -> f64 {
    assert_eq!(references.len(), hypotheses.len(), "Jumlah reference dan hypothesis harus sama!");

    let mut total_matches = [0.0; 4];
    let mut total_possible = [0.0; 4];
    let mut ref_len = 0;
    let mut hyp_len = 0;

    for (r, h) in references.iter().zip(hypotheses.iter()) {
        ref_len += r.len();
        hyp_len += h.len();

        for n in 1..=4 {
            let matches = count_ngram_matches(r, h, n);
            let possible = if h.len() >= n { h.len() - n + 1 } else { 0 };
            total_matches[n - 1] += matches as f64;
            total_possible[n - 1] += possible as f64;
        }
    }

    if hyp_len == 0 {
        return 0.0;
    }

    let mut score = 0.0;
    for n in 0..4 {
        if total_possible[n] == 0.0 || total_matches[n] == 0.0 {
            return 0.0; // Jika tidak ada match sama sekali pada N-gram tertentu, BLEU=0
        }
        score += (total_matches[n] / total_possible[n]).ln();
    }
    score /= 4.0;
    
    // Brevity Penalty
    let bp = if hyp_len > ref_len {
        1.0
    } else {
        (1.0 - (ref_len as f64 / hyp_len as f64)).exp()
    };

    bp * score.exp() * 100.0 // Skala persentase (0 - 100)
}

fn count_ngram_matches(r: &[usize], h: &[usize], n: usize) -> usize {
    if h.len() < n || r.len() < n {
        return 0;
    }

    let mut ref_counts = HashMap::new();
    for i in 0..=r.len() - n {
        let ngram = &r[i..i + n];
        *ref_counts.entry(ngram.to_vec()).or_insert(0) += 1;
    }

    let mut matches = 0;
    let mut hyp_counts = HashMap::new();
    for i in 0..=h.len() - n {
        let ngram = &h[i..i + n];
        *hyp_counts.entry(ngram.to_vec()).or_insert(0) += 1;
    }

    // Clip jumlah kemunculan hypothesis berdasarkan referensi
    for (ngram, count) in hyp_counts {
        if let Some(r_count) = ref_counts.get(&ngram) {
            matches += count.min(*r_count);
        }
    }

    matches
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bleu_perfect_match() {
        let refs = vec![vec![1, 2, 3, 4, 5, 2]];
        let hyps = vec![vec![1, 2, 3, 4, 5, 2]];
        let bleu = calculate_bleu(&refs, &hyps);
        assert!((bleu - 100.0).abs() < 1e-4);
    }

    #[test]
    fn test_bleu_zero_match() {
        let refs = vec![vec![1, 2, 3, 4, 5]];
        let hyps = vec![vec![6, 7, 8, 9, 10]];
        let bleu = calculate_bleu(&refs, &hyps);
        assert_eq!(bleu, 0.0);
    }
}
