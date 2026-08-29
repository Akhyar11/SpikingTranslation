use spiking_translation::data::corpus::StreamingCorpus;
use spiking_translation::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};
use spiking_translation::snn::lif::LifParameters;
use spiking_translation::data::ngram::SparseNGramMemory;
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};
use std::fs::File;

#[derive(Serialize, Deserialize)]
struct SpikingCheckpoint {
    enc_w_e: Array2<f32>,
    enc_w_r: Array2<f32>,
    stcm_w_ce: Array2<f32>,
    stcm_w_cc: Array2<f32>,
    stcm_w_ctx: Array2<f32>,
    stcm_w_self: Array2<f32>,
    dec_w_y: Array2<f32>,
    dec_w_c: Array2<f32>,
    dec_w_r: Array2<f32>,
}

fn print_stats(name: &str, arr: &Array1<f32>) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut sum = 0.0;
    let mut non_zero = 0;
    for &v in arr.iter() {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v;
        if v > 0.0 { non_zero += 1; }
    }
    let mean = sum / arr.len() as f32;
    println!("  [{}] Min: {:.4}, Max: {:.4}, Mean: {:.4}, Active(>0): {}/{}", name, min, max, mean, non_zero, arr.len());
}

fn debug_infer(
    encoder: &SpikingEncoder,
    stcm: &STCM,
    decoder: &SpikingDecoder,
    src_seq: &[usize],
    corpus: &StreamingCorpus,
    ngram_memory: &SparseNGramMemory,
) {
    let k = 20; // neural timesteps
    let max_len = 5; // Batasi 5 kata saja agar log tidak terlalu panjang
    
    println!("\n=======================================================");
    println!("DEBUG INFERENSI: Memulai proses untuk input Sequence");
    println!("Input Asli: '{}'", corpus.decode(src_seq));
    println!("Token IDs : {:?}", src_seq);
    println!("=======================================================\n");

    let mut s_e_prev = vec![Array1::zeros(encoder.w_e.raw_dim()[0]); k];
    let mut u_e_prev = Array1::zeros(encoder.w_e.raw_dim()[0]);
    let mut s_c_prev = vec![Array1::zeros(stcm.w_cc.raw_dim()[0]); k];
    let mut u_c_prev = Array1::zeros(stcm.w_cc.raw_dim()[0]);
    
    println!("---> FASE 1: ENCODER (Memasukkan token bahasa sumber ke SNN)");
    for (t, &token) in src_seq.iter().enumerate() {
        println!("\n  Waktu ke-{} | Memproses Token ID: {} ('{}')", t, token, corpus.decode(&[token]).trim());
        
        // 1. One-hot Input
        let mut s_x = vec![Array1::zeros(encoder.w_e.raw_dim()[1]); k];
        if token < s_x[0].len() {
            for tau in 0..k { s_x[tau][token] = 1.0; }
            println!("  [s_x] Spike Input aktif di indeks {} selama K=20 timestep", token);
        } else {
            println!("  [s_x] WARNING: Token ID {} di luar jangkauan vocabulary encoder!", token);
        }
        
        let (u_e, s_e) = encoder.forward_token(&s_x, &s_e_prev, &u_e_prev);
        let (u_c, s_c) = stcm.forward_source_token(&s_e, &s_c_prev, &u_c_prev);
        
        print_stats("Encoder Output (u_e)", u_e.last().unwrap());
        print_stats("Encoder Spikes (s_e)", s_e.last().unwrap());
        print_stats("STCM Membran   (u_c)", u_c.last().unwrap());
        print_stats("STCM Spikes    (s_c)", s_c.last().unwrap());
        
        s_e_prev = s_e;
        u_e_prev = u_e.last().unwrap().clone();
        s_c_prev = s_c;
        u_c_prev = u_c.last().unwrap().clone();
    }
    
    println!("\n---> FASE 2: DECODER (Menghasilkan terjemahan autoregresif)");
    let mut s_ctx_prev = s_c_prev;
    let mut u_ctx_prev = u_c_prev;
    let mut s_d_prev = vec![Array1::zeros(decoder.w_r.raw_dim()[0]); k];
    let mut u_d_prev = Array1::zeros(decoder.w_r.raw_dim()[0]);
    
    let mut current_token = 2; // <BOS> / <EOS> di BPE (index 2)
    println!("  Mulai Decoder dengan Token Awal: 2 (<BOS>)");

    let vocab_tgt = decoder.w_y.raw_dim()[1];
    let d_d = decoder.w_r.raw_dim()[0];

    for t in 0..max_len {
        println!("\n  --- Iterasi Decoder {} ---", t + 1);
        println!("  Input ke Decoder (Dari kata sebelumnya): {} ('{}')", current_token, corpus.decode(&[current_token]).trim());
        
        let mut s_y_prev = vec![Array1::zeros(vocab_tgt); k];
        if current_token < vocab_tgt {
            for tau in 0..k { s_y_prev[tau][current_token] = 1.0; }
        }
        
        let (u_ctx, s_ctx) = stcm.forward_decoder_token(&s_d_prev, &s_ctx_prev, &u_ctx_prev);
        let (u_d, s_d) = decoder.forward_token(&s_y_prev, &s_ctx, &s_d_prev, &u_d_prev);
        
        print_stats("STCM Context (u_ctx)", u_ctx.last().unwrap());
        print_stats("Decoder Membran(u_d)", u_d.last().unwrap());
        print_stats("Decoder Spikes (s_d)", s_d.last().unwrap());

        let mut neuron_s_sums = vec![0.0; d_d];
        for tau in 0..k {
            for (i, val) in s_d[tau].iter().enumerate() {
                neuron_s_sums[i] += val;
            }
        }
        
        let mut token_scores = vec![0.0; vocab_tgt];
        for v in 0..vocab_tgt {
            let m_v = spiking_translation::snn::architecture::sdr_token_map(v, d_d, 3);
            for &neuron_id in &m_v {
                token_scores[v] += neuron_s_sums[neuron_id];
            }
        }
        
        // Cari Top 3 Sebelum N-Gram
        let mut sorted_scores: Vec<(usize, f32)> = token_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("  Top 3 SDR SNN murni:");
        for i in 0..3 {
            println!("    {}. '{}' (ID:{}) -> Score: {:.4}", i+1, corpus.decode(&[sorted_scores[i].0]).trim(), sorted_scores[i].0, sorted_scores[i].1);
        }

        // Apply NGram
        let candidates = ngram_memory.get_candidates(current_token);
        if !candidates.is_empty() {
            println!("  N-Gram menyarankan {} kandidat (Boost x2.0).", candidates.len());
            for (tok, prob) in candidates {
                if tok < token_scores.len() {
                    token_scores[tok] += prob * 2.0;
                }
            }
        } else {
            println!("  N-Gram TIDAK memiliki saran untuk kata ini.");
        }
        
        let mut best_token = 1;
        let mut max_score = f32::MIN;
        for (i, &s) in token_scores.iter().enumerate() {
            if i < 2 { continue; } 
            if s > max_score {
                max_score = s;
                best_token = i;
            }
        }
        
        println!("  >>> TERPILIH: '{}' (ID:{}) dengan skor akhir: {:.4}", corpus.decode(&[best_token]).trim(), best_token, max_score);
        
        current_token = best_token;
        s_ctx_prev = s_ctx;
        u_ctx_prev = u_ctx.last().unwrap().clone();
        s_d_prev = s_d;
        u_d_prev = u_d.last().unwrap().clone();
        
        if current_token == 2 {
            println!("  [Mencapai Token EOS]");
            break;
        }
    }
}

fn main() {
    let corpus = StreamingCorpus::new("dataset/OpenSubtitles.en-id.en", "dataset/OpenSubtitles.en-id.id");
    let mut ngram_memory = SparseNGramMemory::new();
    ngram_memory.build_from_corpus(&corpus, 75000);
    
    let file = File::open("best_model.json").expect("File best_model.json tidak ditemukan!");
    let checkpoint: SpikingCheckpoint = serde_json::from_reader(file).expect("Parsing gagal");
    
    let lif_params = LifParameters::new(0.9, 0.3, 1.0);
    let beta_seq = 0.1;
    
    let encoder = SpikingEncoder::new(checkpoint.enc_w_e, checkpoint.enc_w_r, lif_params.clone(), beta_seq);
    let stcm = STCM::new(checkpoint.stcm_w_ce, checkpoint.stcm_w_cc, checkpoint.stcm_w_ctx, checkpoint.stcm_w_self, lif_params.clone(), beta_seq);
    let decoder = SpikingDecoder::new(checkpoint.dec_w_y, checkpoint.dec_w_c, checkpoint.dec_w_r, lif_params, beta_seq);
    
    // Hanya debug 1 sequence
    let text = "People.";
    let mut src_indices = corpus.encode(text);
    src_indices.push(corpus.get_eos_id());
    
    debug_infer(&encoder, &stcm, &decoder, &src_indices, &corpus, &ngram_memory);
}
