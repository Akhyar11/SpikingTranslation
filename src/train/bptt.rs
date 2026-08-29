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
    
    for &token in src {
        let mut s_x = vec![Array1::zeros(encoder.w_e.raw_dim()[1]); k];
        if token < s_x[0].len() { s_x[0][token] = 1.0; }
        
        let (u_e, s_e) = encoder.forward_token(&s_x, &s_e_prev, &u_e_prev);
        let (u_c, s_c) = stcm.forward_source_token(&s_e, &s_c_prev, &u_c_prev);
        
        s_e_prev = s_e;
        u_e_prev = u_e.last().unwrap().clone();
        s_c_prev = s_c;
        u_c_prev = u_c.last().unwrap().clone();
    }
    
    // --- 2. DECODER FORWARD PASS ---
    let mut s_ctx_prev = s_c_prev;
    let mut u_ctx_prev = u_c_prev;
    
    let mut loss_gradients = Vec::new();
    let mut u_histories = Vec::new();
    let mut s_y_histories = Vec::new();
    let mut s_d_histories = Vec::new();
    let mut s_ctx_histories = Vec::new();

    for t in 0..seq_length {
        let prev_token = if t == 0 { 2 } else { tgt[t-1] }; // 2 = BOS/EOS
        let mut s_y = vec![Array1::<f32>::zeros(d_in); k];
        if prev_token < d_in {
            for tau in 0..k { s_y[tau][prev_token] = 1.0; }
        }
        
        let (u_ctx, s_ctx) = stcm.forward_decoder_token(&s_d_prev, &s_ctx_prev, &u_ctx_prev);
        let (u_d, s_d) = decoder.forward_token(&s_y, &s_ctx, &s_d_prev, &u_d_prev);
        let (loss_val, dl_ds) = margin_spike_loss(&s_d, tgt[t], c_t, token_map, 2.0);
        total_loss += loss_val;
        
        u_histories.push(u_d.clone());
        s_y_histories.push(s_y);
        s_d_histories.push(s_d_prev.clone());
        s_ctx_histories.push(s_ctx.clone());
        loss_gradients.push(dl_ds);
        
        u_d_prev = u_d.last().unwrap().clone();
        s_d_prev = s_d.clone();
        s_ctx_prev = s_ctx;
        u_ctx_prev = u_ctx.last().unwrap().clone();
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
