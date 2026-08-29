use ndarray::{Array1, Array2, Zip};

/// Menghitung Surrogate Gradient menggunakan fungsi Boxcar (persegi).
/// `z` = U - threshold.
/// `width` = Batas jendela (window). Jika jarak voltase ke threshold < width, gradien = 1.0, sebaliknya 0.0.
pub fn boxcar_surrogate(z: f32, width: f32) -> f32 {
    if z.abs() < width {
        1.0
    } else {
        0.0
    }
}

/// Fungsi pembantu untuk vektor z (ndarray)
pub fn boxcar_surrogate_array(z: &Array1<f32>, width: f32) -> Array1<f32> {
    z.mapv(|val| boxcar_surrogate(val, width))
}

/// Parameter neuron Leaky Integrate-and-Fire (LIF)
#[derive(Clone, Debug)]
pub struct LifParameters {
    pub beta: f32,       // Decay factor (biasanya < 1.0, misal 0.9)
    pub threshold: f32,  // Ambang batas penembakan spike (vartheta)
    pub width: f32,      // Lebar boxcar surrogate gradient
}

impl LifParameters {
    pub fn new(beta: f32, threshold: f32, width: f32) -> Self {
        Self {
            beta,
            threshold,
            width,
        }
    }

    /// Forward pass 1 langkah temporal (neural / sequence time).
    /// Mengembalikan tuple: (u_next, s_next)
    /// Persamaan:
    /// U[t] = beta * U[t-1] + I[t] - threshold * S[t-1]
    /// S[t] = H(U[t] - threshold)
    pub fn forward_step(&self, u_prev: f32, i_t: f32, s_prev: f32) -> (f32, f32) {
        let u_next = self.beta * u_prev + i_t - self.threshold * s_prev;
        let s_next = if u_next >= self.threshold { 1.0 } else { 0.0 };
        (u_next, s_next)
    }

    /// Menghitung output gradien (g) pada status saat ini untuk keperluan backprop
    pub fn surrogate_derivative(&self, u_t: f32) -> f32 {
        boxcar_surrogate(u_t - self.threshold, self.width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxcar_surrogate() {
        let width = 0.5;
        // z dekat dengan 0 (dalam rentang width)
        assert_eq!(boxcar_surrogate(0.1, width), 1.0);
        assert_eq!(boxcar_surrogate(-0.4, width), 1.0);
        
        // z di luar rentang width
        assert_eq!(boxcar_surrogate(0.6, width), 0.0);
        assert_eq!(boxcar_surrogate(-1.0, width), 0.0);
    }

    #[test]
    fn test_lif_forward() {
        let params = LifParameters::new(0.9, 1.0, 0.5);
        
        // Kasus 1: Input kecil, tidak tembak
        let (u1, s1) = params.forward_step(0.0, 0.5, 0.0);
        assert_eq!(u1, 0.5);
        assert_eq!(s1, 0.0);

        // Kasus 2: Akumulasi menyebabkan tembak (0.9 * 0.5 + 0.6 = 1.05)
        let (u2, s2) = params.forward_step(u1, 0.6, s1);
        assert!((u2 - 1.05).abs() < f32::EPSILON);
        assert_eq!(s2, 1.0); // Tembak!

        // Kasus 3: Reset setelah tembak (0.9 * 1.05 + 0.1 - 1.0 * 1.0 = 0.045)
        let (u3, s3) = params.forward_step(u2, 0.1, s2);
        assert!((u3 - 0.045).abs() < f32::EPSILON);
        assert_eq!(s3, 0.0);
    }
}
