use ndarray::{Array1, Zip};

/// Menghitung satu langkah Delta-BPTT mundur (backward pass) pada waktu neural `tau`.
/// Menggabungkan tiga jalur gradien secara bersamaan (Reset-aware):
/// 1. Jalur Spike lokal (e * g)
/// 2. Jalur Membran temporal (beta * d_next)
/// 3. Jalur Reset temporal (-threshold * g * d_next)
/// 
/// Parameter:
/// * `e_t_tau`: Gradien eksternal yang masuk ke $S_t^\tau$ (dari layer atas atau sequence waktu $t+1$)
/// * `g_t_tau`: Nilai dari surrogate gradient function $g(U_t^\tau - \vartheta)$
/// * `delta_next_tau`: Gradien $U$ dari masa depan neural time ($\delta_t^{\tau+1}$)
/// * `beta`: Decay factor membran
/// * `threshold`: Ambang batas penembakan ($\vartheta$)
pub fn delta_bptt_step(
    e_t_tau: f32,
    g_t_tau: f32,
    delta_next_tau: f32,
    beta: f32,
    threshold: f32,
) -> f32 {
    (e_t_tau * g_t_tau) + delta_next_tau * (beta - threshold * g_t_tau)
}

/// Fungsi pembantu untuk vektor (Array1 dari ndarray)
pub fn delta_bptt_step_array(
    e_t_tau: &Array1<f32>,
    g_t_tau: &Array1<f32>,
    delta_next_tau: &Array1<f32>,
    beta: f32,
    threshold: f32,
) -> Array1<f32> {
    let mut delta_t_tau = Array1::zeros(e_t_tau.raw_dim());
    Zip::from(&mut delta_t_tau)
        .and(e_t_tau)
        .and(g_t_tau)
        .and(delta_next_tau)
        .for_each(|d, &e, &g, &d_next| {
            *d = (e * g) + d_next * (beta - threshold * g);
        });
    delta_t_tau
}

/// Menghitung propagasi balik (Backward Boundary) antar-token.
/// Meneruskan gradien dari awal kata saat ini (t, 0) kembali ke akhir kata sebelumnya (t-1, K).
/// Rumus: delta_{t-1}^K <- delta_{t-1}^K + beta_seq * delta_t^0
pub fn delta_bptt_boundary_step(
    delta_prev_k: &mut Array1<f32>,
    delta_t_0: &Array1<f32>,
    beta_seq: f32,
) {
    Zip::from(delta_prev_k)
        .and(delta_t_0)
        .for_each(|d_prev, &d_t0| {
            *d_prev += beta_seq * d_t0;
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_delta_bptt_step_scalar() {
        let beta = 0.9;
        let threshold = 1.0;
        
        // Skenario 1: Neuron tidak menembak (g = 0.0).
        // Gradien lokal terputus, gradien masa depan hanya ter-decay oleh beta.
        let e = 1.0;
        let g = 0.0;
        let d_next = 0.5;
        let expected_1 = (1.0 * 0.0) + 0.5 * (0.9 - 1.0 * 0.0); // 0.45
        let result_1 = delta_bptt_step(e, g, d_next, beta, threshold);
        assert!((result_1 - expected_1).abs() < f32::EPSILON);
        assert_eq!(result_1, 0.45);

        // Skenario 2: Neuron menembak (g = 1.0) dengan boxcar surrogate.
        // Gradien lokal diteruskan. Gradien masa depan dikurangi reset penalty.
        let e = 2.0;
        let g = 1.0;
        let d_next = 0.5;
        let expected_2 = (2.0 * 1.0) + 0.5 * (0.9 - 1.0 * 1.0); // 2.0 + 0.5 * (-0.1) = 1.95
        let result_2 = delta_bptt_step(e, g, d_next, beta, threshold);
        assert!((result_2 - expected_2).abs() < f32::EPSILON);
        assert_eq!(result_2, 1.95);
    }

    #[test]
    fn test_delta_bptt_step_array() {
        let beta = 0.8;
        let threshold = 1.0;
        
        let e = array![1.0, 2.0, 0.0];
        let g = array![0.0, 1.0, 1.0];
        let d_next = array![0.5, 0.5, 0.5];

        let result = delta_bptt_step_array(&e, &g, &d_next, beta, threshold);
        
        // Index 0: 1.0*0 + 0.5*(0.8 - 0) = 0.4
        // Index 1: 2.0*1 + 0.5*(0.8 - 1.0) = 2.0 - 0.1 = 1.9
        // Index 2: 0.0*1 + 0.5*(0.8 - 1.0) = 0.0 - 0.1 = -0.1
        assert!((result[0] - 0.4).abs() < 1e-5);
        assert!((result[1] - 1.9).abs() < 1e-5);
        assert!((result[2] - (-0.1)).abs() < 1e-5);
    }
}
