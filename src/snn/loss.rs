use std::collections::HashSet;
use ndarray::Array1;

/// 5.2 Menghitung Spike Accumulation (A_t) untuk sebuah vocabulary target $v$.
/// Mengkalkulasi jumlah kemunculan spike pada subset indeks neuron $M_v$.
/// $A_t(v) = \sum_{\tau=1}^K \sum_{j \in M_v} S_{d,t,j}^\tau$
/// 
/// * `s_d_t`: history spike dari decoder selama 1 token (K x D_d)
/// * `m_v`: set indeks neuron yang dialokasikan untuk token $v$
pub fn surrogate_derivative(u: f32, threshold: f32, gamma: f32) -> f32 {
    let diff = u - threshold;
    gamma / (1.0 + 5.0 * diff.abs()).powi(2)
}

pub fn compute_spike_score(s_d_t: &[Array1<f32>], m_v: &HashSet<usize>) -> f32 {
    let mut score = 0.0;
    for s_tau in s_d_t {
        for &j in m_v {
            if j < s_tau.len() {
                score += s_tau[j];
            }
        }
    }
    score
}

/// Menghitung skor untuk subset kandidat (C_t) dan mengembalikan token pemenang (argmax).
/// Ini merupakan prediksi token tanpa Softmax dan tanpa dense weight matrix O(V).
pub fn predict_argmax(
    s_d_t: &[Array1<f32>],
    c_t: &HashSet<usize>,
    token_to_neurons: &impl Fn(usize) -> HashSet<usize>, 
) -> (usize, f32) {
    let mut best_token = 0;
    let mut best_score = f32::NEG_INFINITY;
    
    for &v in c_t {
        let m_v = token_to_neurons(v);
        let score = compute_spike_score(s_d_t, &m_v);
        if score > best_score {
            best_score = score;
            best_token = v;
        }
    }
    (best_token, best_score)
}

/// 5.3 Menghitung Margin Spike Loss dan gradien eksternal (Loss) awal untuk $S_{d,t}$
/// $L_t = \max(0, m - A_t^+ + A_t^-)$
pub fn margin_spike_loss(
    s_d_t: &[Array1<f32>],
    y_true: usize,
    c_t: &HashSet<usize>, // Subset Kandidat Token (C_t)
    token_to_neurons: &impl Fn(usize) -> HashSet<usize>,
    margin: f32,
) -> (f32, Vec<Array1<f32>>) {
    // 1. Hitung score target ground truth (A_t^+)
    let m_y = token_to_neurons(y_true);
    let a_plus = compute_spike_score(s_d_t, &m_y);

    // 2. Cari kandidat negatif terbaik (A_t^-), di mana v != y_true
    let mut best_negative_token = None;
    let mut a_minus = f32::NEG_INFINITY;
    for &v in c_t {
        if v == y_true { continue; }
        let score = compute_spike_score(s_d_t, &token_to_neurons(v));
        if score > a_minus {
            a_minus = score;
            best_negative_token = Some(v);
        }
    }

    // Jika C_t kosong atau hanya berisi target, loss = 0
    if best_negative_token.is_none() {
        let k = s_d_t.len();
        let dim = s_d_t[0].len();
        return (0.0, vec![Array1::zeros(dim); k]);
    }
    
    let a_minus = a_minus;
    let best_neg = best_negative_token.unwrap();
    let m_neg = token_to_neurons(best_neg);

    // 3. Hitung nilai Loss L_t
    let loss_val = (margin - a_plus + a_minus).max(0.0);

    // 4. Hitung turunan turunan eksternal dL/dS_d
    // dL/dA^+ = -1, dL/dA^- = 1 (jika margin terlewati, yaitu loss > 0)
    let k = s_d_t.len();
    let dim = s_d_t[0].len();
    let mut dL_dS = vec![Array1::zeros(dim); k];

    if loss_val > 0.0 {
        // Karena turunan terhadap spike akumulasi adalah sama untuk setiap neural time (\tau),
        // gradien disalin merata ke seluruh \tau.
        for tau in 0..k {
            for &j in &m_y {
                if j < dim {
                    dL_dS[tau][j] += -1.0;
                }
            }
            for &j in &m_neg {
                if j < dim {
                    dL_dS[tau][j] += 1.0;
                }
            }
        }
    }

    (loss_val, dL_dS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_margin_spike_loss_and_argmax() {
        let k = 2;
        // Simulasi s_d_t memiliki K=2 neural steps, D_d=4 neuron di layer decoder
        let s_d_t = vec![
            array![1.0, 0.0, 1.0, 0.0],
            array![1.0, 0.0, 0.0, 1.0],
        ];

        // Mock sparse token ke indeks neuron:
        // Token 10 diwakili neuron {0, 1}
        // Token 20 diwakili neuron {2, 3}
        // Token 30 diwakili neuron {0, 3}
        let token_map = |v: usize| -> HashSet<usize> {
            let mut set = HashSet::new();
            match v {
                10 => { set.insert(0); set.insert(1); },
                20 => { set.insert(2); set.insert(3); },
                30 => { set.insert(0); set.insert(3); },
                _ => {}
            }
            set
        };

        // Manual Score Check:
        // Token 10 = s(0,0) + s(0,1) + s(1,0) + s(1,1) = 1 + 0 + 1 + 0 = 2.0
        // Token 20 = s(0,2) + s(0,3) + s(1,2) + s(1,3) = 1 + 0 + 0 + 1 = 2.0
        // Token 30 = s(0,0) + s(0,3) + s(1,0) + s(1,3) = 1 + 0 + 1 + 1 = 3.0
        
        let c_t: HashSet<usize> = vec![10, 20, 30].into_iter().collect();

        // 1. Test Argmax
        let (best_v, best_score) = predict_argmax(&s_d_t, &c_t, &token_map);
        assert_eq!(best_v, 30);
        assert_eq!(best_score, 3.0);

        // 2. Test Loss
        // (y_true = 10, target harusnya 10 (skor 2.0), tapi token 30 punya skor tertinggi (3.0))
        // margin = 1.0. Loss = max(0, 1.0 - 2.0 + 3.0) = 2.0.
        let (loss, grad) = margin_spike_loss(&s_d_t, 10, &c_t, &token_map, 1.0);
        assert_eq!(loss, 2.0);

        // 3. Test Gradien Sparse dengan overlapping neuron
        // Gradien untuk m_y (0, 1) = -1. Gradien untuk m_neg (0, 3) = 1.
        // Index 0 muncul di target DAN negative (overlap), maka: -1 + 1 = 0
        // Membuktikan gradien neuron yang overlapping saling meniadakan.
        assert_eq!(grad[0][0], 0.0);
        assert_eq!(grad[0][1], -1.0);
        assert_eq!(grad[0][2], 0.0);
        assert_eq!(grad[0][3], 1.0);
    }
}
