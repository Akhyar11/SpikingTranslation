use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize, Deserialize};

use crate::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};
use crate::snn::lif::LifParameters;
use crate::data::corpus::StreamingCorpus;

use super::optimizer::Gradients;
use super::bptt::compute_sequence_gradients;

#[derive(Serialize, Deserialize)]
pub struct SpikingCheckpoint {
    pub enc_w_e: Array2<f32>,
    pub enc_w_r: Array2<f32>,
    pub stcm_w_ce: Array2<f32>,
    pub stcm_w_cc: Array2<f32>,
    pub stcm_w_ctx: Array2<f32>,
    pub stcm_w_self: Array2<f32>,
    pub dec_w_y: Array2<f32>,
    pub dec_w_c: Array2<f32>,
    pub dec_w_r: Array2<f32>,
}

pub fn run_training_loop(corpus: &StreamingCorpus) {
    println!("=== Memulai Pelatihan SNN Multithreading (Phase 7) ===");
    
    let vocab_src = corpus.vocab_size();
    let vocab_tgt = corpus.vocab_size();
    
    // Konfigurasi Hyperparameter
    let d_in_src = vocab_src;
    let d_in_tgt = vocab_tgt;
    let d_e = 64;
    let d_c = 128;
    let d_d = 128;

    let lif = LifParameters::new(0.9, 0.3, 1.0);
    let beta_seq = 0.1;
    let learning_rate = 0.01;
    let batch_size = 32;
    let max_samples = 99_000; // Selaras dengan JAX (99k train, 1k test)
    
    // Inisiasi Arsitektur Penuh
    let dist_in = Uniform::new(-0.5, 0.5);
    let dist = Uniform::new(-0.1, 0.1);

    let mut encoder = SpikingEncoder::new(
        Array2::random((d_e, d_in_src), dist_in), 
        Array2::eye(d_e) * 0.9 + Array2::random((d_e, d_e), dist) * 0.1, 
        lif.clone(), 
        beta_seq
    );
    let mut stcm = STCM::new(
        Array2::random((d_c, d_e), dist_in),
        Array2::eye(d_c) * 1.0 + Array2::random((d_c, d_c), dist) * 0.1,
        Array2::random((d_c, d_d), dist_in),
        Array2::eye(d_c) * 1.0 + Array2::random((d_c, d_c), dist) * 0.1,
        lif.clone(),
        0.5
    );
    let mut decoder = SpikingDecoder::new(
        Array2::random((d_d, d_in_tgt), dist_in), 
        Array2::random((d_d, d_c), dist_in), 
        Array2::eye(d_d) * 0.9 + Array2::random((d_d, d_d), dist) * 0.1, 
        lif.clone(), 
        beta_seq
    );

    // Kaiming Init Decoder
    decoder.w_y.mapv_inplace(|_| (rand::random::<f32>() - 0.5) * 0.1);
    decoder.w_c.mapv_inplace(|_| (rand::random::<f32>() - 0.5) * 0.1);
    decoder.w_r.mapv_inplace(|_| (rand::random::<f32>() - 0.5) * 0.1);

    let token_map = |v: usize| -> HashSet<usize> {
        crate::snn::architecture::sdr_token_map(v, d_d, 3)
    };
    let c_t: HashSet<usize> = (1..vocab_tgt).collect();
    
    let mut log_file = OpenOptions::new().create(true).append(true).open("training.log").unwrap();
    writeln!(log_file, "=== Memulai Pelatihan SNN Multithreading Full Scale (500k) ===").unwrap();

    let epochs = 10;
    let total_batches = max_samples / batch_size;
    let mut best_loss = f32::MAX;
    
    for epoch in 1..=epochs {
        println!("Memulai Epoch {}...", epoch);
        writeln!(log_file, "Memulai Epoch {}...", epoch).unwrap();
        let mut total_loss = 0.0;
        let mut batches_processed = 0;
        
        let mut iter = corpus.stream_batches(batch_size, max_samples);
        
        let pb = ProgressBar::new(total_batches as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} [{per_sec} | ETA: {eta}] [Loss: {msg}]")
            .unwrap()
            .progress_chars("##-"));
        
        while let Some((src_batch, tgt_batch)) = iter.next() {
            let batch_results: Vec<(f32, Gradients)> = src_batch.par_iter().zip(tgt_batch.par_iter())
                .map(|(src, tgt)| {
                    compute_sequence_gradients(&encoder, &stcm, &decoder, src, tgt, &c_t, &token_map)
                }).collect();

            let mut batch_loss = 0.0;
            let mut batch_grads = Gradients::zeros(&encoder, &stcm, &decoder);
            
            for (loss, grads) in batch_results {
                batch_loss += loss;
                batch_grads.add(&grads);
            }
            total_loss += batch_loss / src_batch.len() as f32;
            
            // SGD UPDATE (Full Network & Batch Averaged)
            let lr_eff = learning_rate / src_batch.len() as f32;
            
            // 1. Update Encoder
            encoder.w_e -= &(&batch_grads.d_we * lr_eff);
            encoder.w_r -= &(&batch_grads.d_wr_enc * lr_eff);
            
            // 2. Update STCM
            stcm.w_ce -= &(&batch_grads.d_wce * lr_eff);
            stcm.w_cc -= &(&batch_grads.d_wcc * lr_eff);
            stcm.w_ctx -= &(&batch_grads.d_wctx * lr_eff);
            stcm.w_self -= &(&batch_grads.d_wself * lr_eff);
            
            // 3. Update Decoder
            decoder.w_y -= &(&batch_grads.d_wy * lr_eff);
            decoder.w_c -= &(&batch_grads.d_wc * lr_eff);
            decoder.w_r -= &(&batch_grads.d_wr_dec * lr_eff);
            
            batches_processed += 1;
            pb.set_message(format!("{:.4}", total_loss / batches_processed as f32));
            pb.inc(1);
        }
        
        pb.finish_with_message("Epoch Selesai");

        let final_loss = total_loss / batches_processed as f32;
        println!("Epoch {:3} | Final Loss: {:.4}", epoch, final_loss);
        writeln!(log_file, "Epoch {:3} | Final Loss: {:.4}", epoch, final_loss).unwrap();
        
        // Cek Metrik Efisiensi AC Counts
        let ac_count = crate::snn::architecture::GLOBAL_AC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let spike_count = crate::snn::architecture::GLOBAL_SPIKE_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        let neuron_evals = crate::snn::architecture::GLOBAL_NEURON_EVALS.load(std::sync::atomic::Ordering::Relaxed);
        
        let efficiency_msg = format!("  -> Spikes Fired: {} ({:.2}%) | True ACs: {}", spike_count, (spike_count as f32 / neuron_evals as f32) * 100.0, ac_count);
        println!("{}", efficiency_msg);
        writeln!(log_file, "{}", efficiency_msg).unwrap();
        
        if final_loss < best_loss {
            println!("🌟 Loss membaik dari {:.4} ke {:.4}! Menyimpan checkpoint...", best_loss, final_loss);
            writeln!(log_file, "🌟 Loss membaik dari {:.4} ke {:.4}! Menyimpan checkpoint...", best_loss, final_loss).unwrap();
            best_loss = final_loss;
            
            let checkpoint = SpikingCheckpoint {
                enc_w_e: encoder.w_e.clone(),
                enc_w_r: encoder.w_r.clone(),
                stcm_w_ce: stcm.w_ce.clone(),
                stcm_w_cc: stcm.w_cc.clone(),
                stcm_w_ctx: stcm.w_ctx.clone(),
                stcm_w_self: stcm.w_self.clone(),
                dec_w_y: decoder.w_y.clone(),
                dec_w_c: decoder.w_c.clone(),
                dec_w_r: decoder.w_r.clone(),
            };
            
            let f = std::fs::File::create("best_model.json").unwrap();
            serde_json::to_writer(f, &checkpoint).unwrap();
        }
    }
}
