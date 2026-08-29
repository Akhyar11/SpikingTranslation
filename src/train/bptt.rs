use std::collections::HashSet;
use ndarray::{Array1, Array2};
use crate::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};
use crate::snn::loss::margin_spike_loss;
use super::optimizer::Gradients;

pub fn add_outer_product(target: &mut Array2<f32>, u: &Array1<f32>, v: &Array1<f32>) {
    let n = target.nrows();
    let m = target.ncols();
    for j in 0..m {
        let v_j = v[j];
        if v_j != 0.0 { // Sifat Gerbang AND / Sparsity Komputasi Sejati
            for i in 0..n {
                target[[i, j]] += u[i] * v_j;
            }
        }
    }
}

pub fn compute_sequence_gradients(
    encoder: &SpikingEncoder,
    stcm: &STCM,
    decoder: &SpikingDecoder,
    src: &[usize],
    tgt: &[usize],
    c_t: &HashSet<usize>,
    token_map: &impl Fn(usize) -> HashSet<usize>,
) -> (f32, Gradients) {
    let k = 5; 
    let mut grads = Gradients::zeros(encoder, stcm, decoder);
    let mut total_loss = 0.0;

    let d_d = decoder.w_r.raw_dim()[0];
    let d_in = decoder.w_y.raw_dim()[1];
    let d_c = decoder.w_c.raw_dim()[1];
    let seq_length = tgt.len();

    let mut u_d_prev = Array1::<f32>::zeros(d_d);
    let mut s_d_prev = vec![Array1::<f32>::zeros(d_d); k];
    
    // --- 1. ENCODER + STCM FORWARD PASS (Source) ---
    let mut s_e_prev = vec![Array1::zeros(encoder.w_e.raw_dim()[0]); k];
    let mut u_e_prev = Array1::zeros(encoder.w_e.raw_dim()[0]);
    let mut s_c_prev = vec![Array1::zeros(stcm.w_cc.raw_dim()[0]); k];
    let mut u_c_prev = Array1::zeros(stcm.w_cc.raw_dim()[0]);
    let mut s_x = vec![Array1::zeros(encoder.w_e.raw_dim()[1]); k];
    
    for &token in src {
        // Reset s_x efficiently
        for tau in 0..k { s_x[tau].fill(0.0); }
        if token < s_x[0].len() { 
            for tau in 0..k { s_x[tau][token] = 1.0; }
        }
        
        encoder.forward_token_in_place(&s_x, &mut s_e_prev, &mut u_e_prev, None);
        stcm.forward_source_token_in_place(&s_e_prev, &mut s_c_prev, &mut u_c_prev, None);
    }
    
    // --- 2. DECODER FORWARD PASS ---
    let mut s_ctx_prev = s_c_prev;
    let mut u_ctx_prev = u_c_prev;
    
    let mut loss_gradients = Vec::with_capacity(seq_length);
    let mut u_histories = Vec::with_capacity(seq_length);
    let mut s_y_histories = Vec::with_capacity(seq_length);
    let mut s_d_histories = Vec::with_capacity(seq_length);
    let mut s_ctx_histories = Vec::with_capacity(seq_length);
    
    let mut s_y = vec![Array1::<f32>::zeros(d_in); k];
    let mut u_d_hist = vec![Array1::<f32>::zeros(d_d); k];

    for t in 0..seq_length {
        let prev_token = if t == 0 { 2 } else { tgt[t-1] }; // 2 = BOS/EOS
        for tau in 0..k { s_y[tau].fill(0.0); }
        if prev_token < d_in {
            for tau in 0..k { s_y[tau][prev_token] = 1.0; }
        }
        
        // Simpan s_d_prev dan s_ctx_prev SEBELUM di-mutate in-place
        s_d_histories.push(s_d_prev.clone());
        
        stcm.forward_decoder_token_in_place(&s_d_prev, &mut s_ctx_prev, &mut u_ctx_prev, None);
        
        // Simpan s_ctx_prev SETELAH di-mutate (menjadi s_ctx_t)
        s_ctx_histories.push(s_ctx_prev.clone());
        
        decoder.forward_token_in_place(&s_y, &s_ctx_prev, &mut s_d_prev, &mut u_d_prev, Some(&mut u_d_hist));
        
        let (loss_val, dl_ds) = margin_spike_loss(&s_d_prev, tgt[t], c_t, token_map, 2.0);
        total_loss += loss_val;
        
        u_histories.push(u_d_hist.clone());
        s_y_histories.push(s_y.clone());
        loss_gradients.push(dl_ds);
    }

    let mut delta_next_k = Array1::<f32>::zeros(d_d);
    for t in (0..seq_length).rev() {
        let u_history = &u_histories[t];
        let dl_ds = &loss_gradients[t];
        let s_y = &s_y_histories[t];
        let s_d_p = &s_d_histories[t];
        let s_ctx = &s_ctx_histories[t];
        
        let mut delta_tau = delta_next_k.clone();
        for tau in (0..k).rev() {
            let g = u_history[tau].mapv(|u| {
                let diff = u - decoder.lif_params.threshold;
                1.0 / (1.0 + 5.0 * diff.abs()).powi(2)
            });
            
            let mut delta_tau_prev = Array1::<f32>::zeros(d_d);
            for j in 0..d_d {
                delta_tau_prev[j] = dl_ds[tau][j] * g[j] + delta_tau[j] * (decoder.lif_params.beta - decoder.lif_params.threshold * g[j]);
            }
            delta_tau = delta_tau_prev;
            
            add_outer_product(&mut grads.d_wy, &delta_tau, &s_y[tau]);
            add_outer_product(&mut grads.d_wc, &delta_tau, &s_ctx[tau]);
            add_outer_product(&mut grads.d_wr_dec, &delta_tau, &s_d_p[tau]);
        }
        delta_next_k = delta_tau * decoder.beta_seq;
    }
    
    (total_loss, grads)
}
