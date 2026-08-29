use ndarray::{Array1, Array2};
use std::collections::HashSet;
use super::lif::LifParameters;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static GLOBAL_AC_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static GLOBAL_SPIKE_COUNT: AtomicUsize = AtomicUsize::new(0);
pub static GLOBAL_NEURON_EVALS: AtomicUsize = AtomicUsize::new(0);

/// Fungsi inti komputasi Neuromorphic (Addition-Only)
/// Menggantikan W.dot(S) yang berupa MACs menjadi penjumlahan murni indeks aktif.
pub fn sparse_addition(w: &Array2<f32>, s: &Array1<f32>) -> Array1<f32> {
    let mut result = Array1::zeros(w.nrows());
    let mut active_count = 0;
    
    for (j, &val) in s.iter().enumerate() {
        if val > 0.5 { // Spike aktif
            active_count += 1;
            for i in 0..w.nrows() {
                result[i] += w[[i, j]]; // Pure addition
            }
        }
    }
    
    // Track Metrics
    GLOBAL_AC_COUNT.fetch_add(active_count * w.nrows(), Ordering::Relaxed);
    GLOBAL_SPIKE_COUNT.fetch_add(active_count, Ordering::Relaxed);
    GLOBAL_NEURON_EVALS.fetch_add(s.len(), Ordering::Relaxed);
    
    result
}

/// A. Spiking Encoder (Source)
pub struct SpikingEncoder {
    pub w_e: Array2<f32>,
    pub w_r: Array2<f32>,
    pub lif_params: LifParameters,
    pub beta_seq: f32,
}

impl SpikingEncoder {
    pub fn new(w_e: Array2<f32>, w_r: Array2<f32>, lif_params: LifParameters, beta_seq: f32) -> Self {
        Self { w_e, w_r, lif_params, beta_seq }
    }

    /// Forward pass untuk satu token penuh (melalui K neural timesteps).
    pub fn forward_token(
        &self,
        s_x_t: &[Array1<f32>],     // [K] array of input spikes
        s_prev_t: &[Array1<f32>],  // [K] array of recurrent spikes dari token sebelumnya
        u_prev_k: &Array1<f32>,    // State membran dari akhir token sebelumnya
    ) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let k = s_x_t.len();
        let mut s_history = Vec::with_capacity(k);
        let mut u_history = Vec::with_capacity(k);

        // Boundary continuity (t, 0) <- (t-1, K)
        let mut u_current = u_prev_k * self.beta_seq;
        let mut s_current = Array1::zeros(u_prev_k.raw_dim());

        for tau in 0..k {
            // I_e,t^tau = W_e * S_x_t^tau + W_r * S_e,t-1^tau
            let i_e = sparse_addition(&self.w_e, &s_x_t[tau]) + sparse_addition(&self.w_r, &s_prev_t[tau]);
            
            // U_e,t^tau = beta * U_e,t^{tau-1} + I - threshold * S^{tau-1}
            u_current = &u_current * self.lif_params.beta + &i_e - &s_current * self.lif_params.threshold;
            
            // S_e,t^tau = H(U - threshold)
            s_current = u_current.mapv(|u| if u >= self.lif_params.threshold { 1.0 } else { 0.0 });
            
            u_history.push(u_current.clone());
            s_history.push(s_current.clone());
        }

        (u_history, s_history)
    }
}

/// B & C. Spiking Temporal Context Memory (STCM)
pub struct STCM {
    pub w_ce: Array2<f32>,   // Encoder to STCM
    pub w_cc: Array2<f32>,   // Source-side recurrence
    pub w_ctx: Array2<f32>,  // Decoder to STCM (dynamic context)
    pub w_self: Array2<f32>, // Decoder-side recurrence
    pub lif_params: LifParameters,
    pub beta_seq: f32,
}

impl STCM {
    pub fn new(w_ce: Array2<f32>, w_cc: Array2<f32>, w_ctx: Array2<f32>, w_self: Array2<f32>, lif_params: LifParameters, beta_seq: f32) -> Self {
        Self { w_ce, w_cc, w_ctx, w_self, lif_params, beta_seq }
    }

    /// B. STCM (Source-side Context Building)
    pub fn forward_source_token(
        &self,
        s_e_t: &[Array1<f32>],
        s_c_prev_t: &[Array1<f32>],
        u_prev_k: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let k = s_e_t.len();
        let mut s_history = Vec::with_capacity(k);
        let mut u_history = Vec::with_capacity(k);

        let mut u_current = u_prev_k * self.beta_seq;
        let mut s_current = Array1::zeros(u_prev_k.raw_dim());

        for tau in 0..k {
            let i_c = sparse_addition(&self.w_ce, &s_e_t[tau]) + sparse_addition(&self.w_cc, &s_c_prev_t[tau]);
            u_current = &u_current * self.lif_params.beta + &i_c - &s_current * self.lif_params.threshold;
            s_current = u_current.mapv(|u| if u >= self.lif_params.threshold { 1.0 } else { 0.0 });
            
            u_history.push(u_current.clone());
            s_history.push(s_current.clone());
        }
        (u_history, s_history)
    }

    /// C. STCM (Decoder-side Dynamic Context)
    pub fn forward_decoder_token(
        &self,
        s_d_prev_t: &[Array1<f32>],
        s_ctx_prev_t: &[Array1<f32>],
        u_prev_k: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let k = s_d_prev_t.len();
        let mut s_history = Vec::with_capacity(k);
        let mut u_history = Vec::with_capacity(k);

        let mut u_current = u_prev_k * self.beta_seq;
        let mut s_current = Array1::zeros(u_prev_k.raw_dim());

        for tau in 0..k {
            let i_ctx = sparse_addition(&self.w_ctx, &s_d_prev_t[tau]) + sparse_addition(&self.w_self, &s_ctx_prev_t[tau]);
            u_current = &u_current * self.lif_params.beta + &i_ctx - &s_current * self.lif_params.threshold;
            s_current = u_current.mapv(|u| if u >= self.lif_params.threshold { 1.0 } else { 0.0 });
            
            u_history.push(u_current.clone());
            s_history.push(s_current.clone());
        }
        (u_history, s_history)
    }
}

/// D. Spiking Decoder (Target)
pub struct SpikingDecoder {
    pub w_y: Array2<f32>,
    pub w_c: Array2<f32>,
    pub w_r: Array2<f32>,
    pub lif_params: LifParameters,
    pub beta_seq: f32,
}

impl SpikingDecoder {
    pub fn new(w_y: Array2<f32>, w_c: Array2<f32>, w_r: Array2<f32>, lif_params: LifParameters, beta_seq: f32) -> Self {
        Self { w_y, w_c, w_r, lif_params, beta_seq }
    }

    pub fn forward_token(
        &self,
        s_y_prev_t: &[Array1<f32>],
        s_ctx_t: &[Array1<f32>],
        s_d_prev_t: &[Array1<f32>],
        u_prev_k: &Array1<f32>,
    ) -> (Vec<Array1<f32>>, Vec<Array1<f32>>) {
        let k = s_ctx_t.len();
        let mut s_history = Vec::with_capacity(k);
        let mut u_history = Vec::with_capacity(k);

        let mut u_current = u_prev_k * self.beta_seq;
        let mut s_current = Array1::zeros(u_prev_k.raw_dim());

        for tau in 0..k {
            let i_d = sparse_addition(&self.w_y, &s_y_prev_t[tau]) + sparse_addition(&self.w_c, &s_ctx_t[tau]) + sparse_addition(&self.w_r, &s_d_prev_t[tau]);
            u_current = &u_current * self.lif_params.beta + &i_d - &s_current * self.lif_params.threshold;
            s_current = u_current.mapv(|u| if u >= self.lif_params.threshold { 1.0 } else { 0.0 });
            
            u_history.push(u_current.clone());
            s_history.push(s_current.clone());
        }
        (u_history, s_history)
    }
}

/// Sparse Distributed Representation (SDR) Token Mapping
/// Memetakan token ID (hingga ratusan ribu) ke kombinasi spesifik neuron di layer berdimensi kecil (d_d)
pub fn sdr_token_map(v: usize, d_d: usize, num_active: usize) -> HashSet<usize> {
    let mut set = HashSet::new();
    let mut seed = (v as u64).wrapping_mul(1234567891).wrapping_add(987654321);
    for _ in 0..num_active {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        set.insert((seed % d_d as u64) as usize);
    }
    // Jika karena hash collision (sangat jarang) elemen kurang dari num_active, 
    // kita biarkan saja agar deterministik dan cepat.
    set
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_end_to_end_architecture_shapes() {
        let k = 4;
        let d_in = 16;
        let d_e = 32;
        let d_c = 48;
        let d_d = 64;

        let lif = LifParameters::new(0.9, 1.0, 1.0);
        
        let encoder = SpikingEncoder::new(
            Array2::zeros((d_e, d_in)), 
            Array2::zeros((d_e, d_e)), 
            lif.clone(), 
            0.5
        );
        
        let stcm = STCM::new(
            Array2::zeros((d_c, d_e)),
            Array2::zeros((d_c, d_c)),
            Array2::zeros((d_c, d_d)),
            Array2::zeros((d_c, d_c)),
            lif.clone(),
            0.5
        );

        let decoder = SpikingDecoder::new(
            Array2::zeros((d_d, d_in)), // asumsi target embedding punya dim yang sama dengan source
            Array2::zeros((d_d, d_c)),
            Array2::zeros((d_d, d_d)),
            lif.clone(),
            0.5
        );

        // Dummy data
        let s_x = vec![Array1::zeros(d_in); k];
        let s_e_prev = vec![Array1::zeros(d_e); k];
        let u_e_prev = Array1::zeros(d_e);

        // Encoder Forward
        let (u_e, s_e) = encoder.forward_token(&s_x, &s_e_prev, &u_e_prev);
        assert_eq!(s_e.len(), k);
        assert_eq!(s_e[0].len(), d_e);

        // STCM Source Forward
        let s_c_prev = vec![Array1::zeros(d_c); k];
        let u_c_prev = Array1::zeros(d_c);
        let (u_c, s_c) = stcm.forward_source_token(&s_e, &s_c_prev, &u_c_prev);
        assert_eq!(s_c.len(), k);
        assert_eq!(s_c[0].len(), d_c);

        // STCM Decoder Forward
        let s_d_prev = vec![Array1::zeros(d_d); k];
        let s_ctx_prev = vec![Array1::zeros(d_c); k]; // dari S_c_T_x
        let u_ctx_prev = Array1::zeros(d_c);
        let (u_ctx, s_ctx) = stcm.forward_decoder_token(&s_d_prev, &s_ctx_prev, &u_ctx_prev);
        assert_eq!(s_ctx.len(), k);
        assert_eq!(s_ctx[0].len(), d_c);

        // Decoder Forward
        let s_y_prev = vec![Array1::zeros(d_in); k];
        let u_d_prev = Array1::zeros(d_d);
        let (u_d, s_d) = decoder.forward_token(&s_y_prev, &s_ctx, &s_d_prev, &u_d_prev);
        assert_eq!(s_d.len(), k);
        assert_eq!(s_d[0].len(), d_d);
    }
}
