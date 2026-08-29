# Audit Implementasi Matematika SNN (SpikingTranslation)

Dokumen ini adalah daftar periksa (checklist) untuk memvalidasi apakah kode sumber (Rust & JAX) benar-benar mematuhi hukum kalkulus *Backpropagation Through Time* (BPTT) dan *Surrogate Gradient* yang telah kita bedah secara manual.

## 1. Fase Forward & Inferensi
- `[x]` **Validasi Pembacaan Spike di Decoder (Kritis)**: 
  - Masalah: Saat ini logika evaluasi/inferensi membaca skor dari akumulasi Membran Potensial (`u_d`), sedangkan algoritma Loss dilatih menggunakan akumulasi Spike (`s_d`). 
  - Tindakan: Cek `src/eval.rs` (Rust) dan `infer.py` (JAX). Pastikan variabel yang dijumlahkan untuk memilih token adalah `s_d` (Spike), BUKAN `u_d`.
- `[x]` **Thresholding & Reset**: Pastikan Soft-Reset hanya mengurangi `u - threshold` tepat di saat `u >= threshold` (Spike = 1).

## 2. Fase Backward & Gradient (Rust: `bptt.rs`)
- `[x]` **Validasi Transpose Input ($S^T$) pada Update Bobot ($\Delta W$)**: 
  - Cek `src/train/bptt.rs`. 
  - Saat menghitung gradien bobot (misal `grad_w_y`), pastikan menggunakan fungsi turunan *outer product* antara Input yang masuk dan Error lokal (misal: perkalian matriks yang melibatkan `.t()` dari vektor input).
- `[x]` **Validasi Transpose Bobot ($W^T$) pada Oper Error Mundur**: 
  - Cek bagaimana `dL_dS` (Error) dialirkan mundur dari Decoder $\rightarrow$ STCM $\rightarrow$ Encoder. 
  - Pastikan menggunakan perkalian titik (*dot product*) dengan matriks bobot yang di-Transpose (misal: `w_c.t().dot(...)`). (DITANGGUHKAN: Mendelegasikan full gradient graph ke JAX/Kaggle).
- `[x]` **Validasi Multiplier (Surrogate Gradient x Error)**:
  - Cek fungsi kalkulasi BPTT. Pastikan turunan dari voltase (Fungsi `surrogate_derivative`) secara eksplisit dikalikan (mutiplikasi skalar/element-wise) dengan Error yang diturunkan dari atas.

## 3. Fase JAX / Kaggle Porting (`train.py`)
- `[x]` **Validasi @jax.custom_vjp**: Pastikan `spike_fn_bwd` benar-benar mengeksekusi rumus Fast Sigmoid: `gamma / (1.0 + 5.0 * abs(u - threshold))^2` dan mengalikannya dengan gradien bawaan `g`.
- `[x]` **Penyelarasan Inferensi JAX**: Pastikan skor token diambil dari `s_d_sum` bukan `u_d` di dalam `infer.py` agar sejajar dengan `margin_spike_loss`.

## 4. Efisiensi & Sparsity
- `[x]` **Validasi Sparsity (Gerbang AND)**: Pastikan di Rust, jika nilai Input Spike adalah 0, maka baris bobot yang berkorespondensi benar-benar tidak dikalkulasi gradiennya (menghasilkan +0.0), sehingga `w` asli tidak berubah dan menghemat memori.
