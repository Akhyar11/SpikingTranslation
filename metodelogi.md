Ya. Saya akan sederhanakan menjadi **satu hipotesis utama dengan 3 RQ**, tetapi metodologinya tetap lengkap sehingga eksperimennya cukup kuat untuk paper.

# Proposed Research Methodology

## Memory-Augmented Fully Spiking Seq2Seq for Parameter-Efficient Machine Translation

---

# 1. Research Objective

Penelitian ini bertujuan menginvestigasi apakah **fully spiking Seq2Seq dengan sekitar 1M trainable parameters** dapat digunakan untuk machine translation, dan apakah **sparse N-gram memory** dapat meningkatkan kapasitas translation tanpa memperbesar neural parameter secara signifikan.

Konsep dasarnya:

$$
\boxed{
\text{Small Fully-Spiking Neural Core}
+
\text{Large Sparse N-gram Memory}
}
$$

Neural core menangani:

$$
\text{temporal representation}
+
\text{sequence transformation}
$$

sedangkan N-gram memory menyediakan:

$$
\text{lexical/contextual retrieval}
$$

Dengan demikian kapasitas sistem tidak seluruhnya harus disimpan dalam parameter neural.

---

# 2. Research Questions

Penelitian dibatasi menjadi tiga pertanyaan.

### RQ1 — Spiking Seq2Seq Capability

> **Can a fully spiking Seq2Seq model with approximately 1M trainable parameters perform effective machine translation?**

Tujuan:

Menguji apakah model SNN kecil dapat melakukan translation tanpa bergantung pada dense neural output seperti Softmax.

---

### RQ2 — N-gram Memory

> **Does sparse N-gram memory improve the translation performance of a parameter-efficient fully spiking Seq2Seq model?**

Eksperimen utama:

$$
\boxed{SNN}
$$

vs.

$$
\boxed{SNN+N\text{-}gram}
$$

Parameter neural kedua model dibuat hampir sama.

Dengan demikian peningkatan performa dapat dikaitkan dengan external memory, bukan peningkatan model capacity.

---

### RQ3 — Computational Efficiency

> **What is the trade-off between translation quality and computational efficiency of the proposed memory-augmented SNN?**

Evaluasi:

$$
BLEU
$$

$$
chrF
$$

$$
Parameters
$$

$$
Latency/token
$$

$$
Tokens/sec
$$

$$
Spike\ rate
$$

---

# 3. Research Hypothesis

Hipotesis dibuat sederhana.

### H1

$$
BLEU(SNN)>0
$$

dan model mampu mempelajari mapping source-target secara meaningful.

### H2

$$
\boxed{
BLEU(SNN+N\text{-}gram)
>
BLEU(SNN)
}
$$

dengan parameter neural yang relatif sama.

### H3

Model memory-augmented memberikan trade-off yang baik antara:

$$
\text{translation quality}
$$

dan:

$$
\text{computational cost}
$$

---

# 4. Overall Architecture

Arsitektur utama:

```text
 Source Sentence
      │
      ▼
Subword Tokenization (BPE)
      │
      ▼
 Spike Encoding
      │
      ├─────────────────────┐
      │                     │
      ▼                     ▼
Spiking Encoder       N-gram Memory
      │                     │
      │                     ▼
      │              Sparse Memory Spikes
      │                     │
      └──────────┬──────────┘
                 ▼
        Temporal Spiking State
                 │
                 ▼
        Spiking Decoder
                 │
                 ▼
       Sparse Spike Output
                 │
                 ▼
       Candidate Vocabulary
                 │
                 ▼
         Spike Accumulation
                 │
                 ▼
              argmax
                 │
                 ▼
          Target Token
```

Tidak terdapat Softmax pada jalur inference utama.

---

# 5. Sequence Representation

Source:

$$
X=(x_1,x_2,\ldots,x_{T_x})
$$

Target:

$$
Y=(y_1,y_2,\ldots,y_{T_y})
$$

Model memiliki dua dimensi temporal:

### Token time

$$
t=1,\ldots,T
$$

### Neural time

$$
\tau=1,\ldots,K
$$

Dengan demikian:

$$
S_t^\tau
$$

berarti spike neuron pada token timestep \(t\) dan neural timestep \(\tau\).

---

# 6. Spike Encoding

Token tidak langsung diberikan sebagai dense embedding ke ANN.

Untuk token \(x_t\), diperoleh spike representation:

$$
S_{x_t}^{\tau}
\in
\{0,1\}^{D}
$$

sehingga:

$$
x_t
\rightarrow
\left[
S_{x_t}^{1},
S_{x_t}^{2},
\ldots,
S_{x_t}^{K}
\right]
$$

Encoding dapat menggunakan sparse learned embedding-to-spike mapping.

Parameter embedding tetap dihitung sebagai bagian:

$$
P_{\mathrm{trainable}}
$$

---

# 7. N-gram Memory

Untuk setiap token position:

$$
g_t^{(n)}
=
(x_{t-n+1},\ldots,x_t)
$$

Untuk:

$$
n\in\{1,2,3,4\}
$$

memory menyimpan mapping:

$$
M_n(g_t^{(n)})
\rightarrow
\text{candidate target representation}
$$

Combined memory:

$$
M(g_t)
=
\bigcup_{n=1}^{4}M_n(g_t^{(n)})
$$

---

# 8. Sparse Memory Representation

Memory tidak disimpan sebagai dense matrix:

$$
M\in\mathbb R^{N\times D}
$$

melainkan sparse event representation:

$$
M(g)
=
\{
(j_1,\tau_1),
(j_2,\tau_2),
\ldots
\}
$$

Sehingga hanya neuron yang aktif yang diproses.

Memory injection:

$$
\boxed{
I_{\mathrm{mem},t}^{\tau}
=
\sum_{n}
\alpha_n
W_n
S_{\mathrm{mem},t}^{(n),\tau}
}
$$

---

# 9. Spiking Encoder

Encoder menggunakan LIF recurrent neurons.

Membrane:

$$
\boxed{
U_{e,t}^{\tau}
=
\beta_eU_{e,t}^{\tau-1}
+
I_{e,t}^{\tau}
-
\vartheta_eS_{e,t}^{\tau-1}
}
$$

Input:

$$
I_{e,t}^{\tau}
=
W_eS_{x_t}^{\tau}
+
W_rS_{e,t-1}^{\tau}
+
I_{\mathrm{mem},t}^{\tau}
$$

Spike:

$$
\boxed{
S_{e,t}^{\tau}
=
H(U_{e,t}^{\tau}-\vartheta_e)
}
$$

---

# 10. Temporal Encoder State

Encoder tidak diringkas menjadi satu dense vector.

State:

$$
H_e=
\left\{
S_{e,t}^{\tau}
\right\}
$$

dengan:

$$
t=1,\ldots,T_x
$$

dan:

$$
\tau=1,\ldots,K
$$

Sehingga informasi temporal tetap berbentuk spike sequence.

---

# 11. Spiking Decoder

Decoder menerima target sebelumnya dan encoder state.

$$
I_{d,t}^{\tau}
=
W_dS_{y_{t-1}}^{\tau}
+
W_cH_{e,t}^{\tau}
+
W_{dr}S_{d,t-1}^{\tau}
$$

Membrane:

$$
\boxed{
U_{d,t}^{\tau}
=
\beta_dU_{d,t}^{\tau-1}
+
I_{d,t}^{\tau}
-
\vartheta_dS_{d,t}^{\tau-1}
}
$$

Spike:

$$
\boxed{
S_{d,t}^{\tau}
=
H(U_{d,t}^{\tau}-\vartheta_d)
}
$$

---

# 12. Sparse Vocabulary Output

Full vocabulary projection dihilangkan.

Daripada:

$$
W_{out}\in
\mathbb R^{V\times D}
$$

digunakan candidate set:

$$
C_t\subset V
$$

dengan:

$$
|C_t|\ll V
$$

Candidate diperoleh dari N-gram memory dan mekanisme vocabulary retrieval.

---

# 13. Spike-Based Token Score

Untuk kandidat \(v\):

$$
M_v=
\{j_1,j_2,\ldots,j_r\}
$$

score:

$$
\boxed{
A_t(v)
=
\sum_{\tau=1}^{K}
\sum_{j\in M_v}
S_{d,t,j}^{\tau}
}
$$

Prediksi:

$$
\boxed{
\hat y_t
=
\arg\max_{v\in C_t}
A_t(v)
}
$$

Tidak diperlukan:

$$
Softmax
$$

untuk inference.

---

# 14. Training Objective

Ground-truth token:

$$
y_t
$$

Positive score:

$$
A_t^+=A_t(y_t)
$$

Negative score:

$$
A_t^-=
\max_{v\in C_t,v\neq y_t}A_t(v)
$$

Margin loss:

$$
\boxed{
L_t=
\max
\left(
0,
m-A_t^++A_t^-
\right)
}
$$

Sequence loss:

$$
\boxed{
L=
\sum_{t=1}^{T_y}L_t
}
$$

---

# 15. Surrogate Gradient

Forward menggunakan:

$$
S=H(U-\vartheta)
$$

tetapi backward menggunakan:

$$
\frac{\partial S}{\partial U}
\approx
g(U-\vartheta)
$$

Misalnya menggunakan Rectangular Surrogate Mask (Boxcar Gradient):

$$
\boxed{
g(z)=
\begin{cases}
1, & |z| < w \\
0, & \text{lainnya}
\end{cases}
}
$$

di mana $w=1.0$.

---

# 16. Delta-BPTT

Definisikan:

$$
\delta_t^\tau
=
\frac{\partial L}
{\partial U_t^\tau}
$$

Dengan:

$$
U_t^{\tau+1}
=
\beta U_t^\tau
+
I_t^{\tau+1}
-
\vartheta S_t^\tau
$$

maka:

$$
\frac{\partial U_t^{\tau+1}}
{\partial U_t^\tau}
=
\beta
-
\vartheta g_t^\tau
$$

sehingga:

$$
\boxed{
\delta_t^\tau
=
\frac{\partial L}{\partial S_t^\tau}
g_t^\tau
+
\delta_t^{\tau+1}
(\beta-\vartheta g_t^\tau)
}
$$

Ini menjadi mekanisme Delta-BPTT utama.

---

# 17. Weight Gradient

Jika:

$$
I_t^\tau=WS_t^\tau
$$

maka:

$$
\boxed{
\frac{\partial L}{\partial W}
=
\sum_t\sum_\tau
\delta_t^\tau
(S_t^\tau)^T
}
$$

Untuk weight \(W_{ij}\):

$$
\boxed{
\frac{\partial L}{\partial W_{ij}}
=
\sum_{t,\tau}
\delta_{i,t}^{\tau}
S_{j,t}^{\tau}
}
$$

Update:

$$
\boxed{
W\leftarrow
W-\eta
\frac{\partial L}{\partial W}
}
$$

---

# 18. Training Procedure

Setiap training sample:

```text
Source X
   ↓
N-gram retrieval
   ↓
Spike encoding
   ↓
Spiking Encoder
   ↓
Temporal encoder state
   ↓
Spiking Decoder
   ↓
Spike output
   ↓
Sparse candidate retrieval
   ↓
Spike score
   ↓
Margin loss
   ↓
Delta-BPTT
   ↓
Parameter update
```

Training menggunakan teacher forcing:

$$
input_t=y_{t-1}
$$

---

# 19. Inference Procedure

Pada inference:

$$
y_0=<BOS>
$$

kemudian:

$$
\hat y_t
=
\arg\max_{v\in C_t}
A_t(v)
$$

Predicted token digunakan kembali:

$$
\hat y_t
\rightarrow
\hat y_{t+1}
$$

hingga:

$$
\hat y_t=<EOS>
$$

Tidak ada ground-truth target yang diberikan selama inference.

---

# 20. Experimental Design

Penelitian tidak perlu menguji terlalu banyak model.

Cukup dua model utama:

## Baseline

$$
\boxed{
SNN
}
$$

Fully spiking Seq2Seq sekitar 1M parameter (dicapai dengan menerapkan Subword Tokenization / BPE untuk membatasi ukuran matriks Vocabulary secara ketat menjadi ~4000 token).

Tanpa N-gram memory.

---

## Proposed

$$
\boxed{
SNN+N\text{-}gram
}
$$

Arsitektur sama, tetapi ditambahkan sparse N-gram memory.

Parameter neural dijaga:

$$
P_{SNN}
\approx
P_{SNN+NG}
\approx1M
$$

Perbedaan utamanya:

$$
\boxed{
N\text{-}gram\ external\ memory
}
$$

---

# 21. Dataset Protocol

Dataset dibagi:

$$
D=
D_{train}
\cup
D_{validation}
\cup
D_{test}
$$

Misalnya:

$$
80/10/10
$$

atau official dataset split.

Yang sangat penting:

$$
\boxed{
M_{ngram}=Build(D_{train})
}
$$

N-gram memory **tidak boleh dibangun dari validation/test data**.

---

# 22. Training Configuration

Parameter yang dilaporkan:

$$
P_{\mathrm{trainable}}
$$

$$
D
$$

$$
K
$$

$$
\beta
$$

$$
\vartheta
$$

$$
\eta
$$

$$
batch\ size
$$

$$
epochs
$$

$$
optimizer
$$

$$
n\text{-}gram\ order
$$

dan:

$$
|C_t|
$$

---

# 23. Model Selection

Training menggunakan:

$$
D_{train}
$$

Validation digunakan untuk memilih checkpoint:

$$
\boxed{
Checkpoint^*
=
\arg\max BLEU_{validation}
}
$$

atau berdasarkan validation loss.

Test set tidak digunakan untuk pemilihan hyperparameter.

---

# 24. RQ1 Experiment

### Tujuan

Menguji apakah:

$$
\boxed{
\sim1M\text{-parameter fully SNN}
}
$$

dapat melakukan machine translation.

Model:

$$
SNN
$$

Metrics:

$$
BLEU
$$

$$
chrF
$$

$$
Token\ Accuracy
$$

Tidak perlu terlalu banyak metric.

---

# 25. RQ2 Experiment

Ini adalah **eksperimen inti penelitian**.

Bandingkan:

$$
SNN
$$

dengan:

$$
SNN+N\text{-}gram
$$

Semua kondisi lain sama:

$$
Architecture
$$

$$
Training\ epochs
$$

$$
Dataset
$$

$$
Optimizer
$$

$$
Parameter\ budget
$$

Yang berbeda hanya:

$$
N\text{-}gram\ memory
$$

Kemudian:

$$
\Delta BLEU
=
BLEU_{SNN+NG}
-
BLEU_{SNN}
$$

Jika:

$$
\Delta BLEU>0
$$

maka ada bukti bahwa external N-gram memory membantu.

---

# 26. RQ3 Experiment

Untuk efisiensi, ukur kedua model:

$$
SNN
$$

dan:

$$
SNN+NG
$$

pada hardware yang sama.

Ukur:

### Latency

$$
Latency/sentence
$$

### Throughput

$$
Tokens/sec
=
\frac{N_{generated}}
{T}
$$

### Spike rate

$$
\rho=
\frac{
\sum S
}{
N_{neuron}N_{timestep}
}
$$

### Spike/token

$$
Spike/Token=
\frac{N_{spike}}
{N_{generated}}
$$

### Memory

$$
RAM
$$

---

# 27. CPU Benchmark

Karena salah satu motivasi penelitianmu adalah computational efficiency, inference benchmark dilakukan:

$$
\boxed{CPU-only}
$$

Tidak menggunakan GPU.

Konfigurasi hardware harus dilaporkan:

```text
CPU:
RAM:
OS:
Threads:
Runtime:
Compiler:
```

Jumlah thread harus sama untuk semua model.

---

# 28. Benchmark Protocol

Sebelum pengukuran:

$$
N_{warmup}=10
$$

Kemudian:

$$
N_{benchmark}=30
$$

atau jumlah yang sesuai.

Laporan:

$$
mean\pm std
$$

untuk:

$$
Latency
$$

dan:

$$
Tokens/sec
$$

---

# 29. Translation Evaluation

Pada test set:

$$
\hat Y_i
=
Model(X_i)
$$

kemudian dibandingkan dengan:

$$
Y_i
$$

Metrics:

$$
\boxed{BLEU}
$$

dan:

$$
\boxed{chrF}
$$

BLEU menjadi metric utama.

chrF menjadi metric pendukung.

---

# 30. Parameter Reporting

Wajib membedakan:

$$
\boxed{
P_{trainable}
}
$$

dengan:

$$
\boxed{
P_{memory}
}
$$

Misalnya:

$$
P_{trainable}=1.03M
$$

dan:

$$
M_{ngram}=20M\ entries
$$

Jangan menyebut sistem secara sederhana sebagai "1M parameters" tanpa menjelaskan external memory.

---

# 31. Efisiensi Tidak Hanya Berdasarkan Spike

Jangan mengatakan:

> SNN lebih hemat energi karena spike lebih sedikit.

Itu belum terbukti.

Yang dapat diklaim secara langsung:

$$
Spike\ sparsity
$$

Sedangkan energy efficiency harus diukur:

$$
Energy/token
$$

jika hardware measurement tersedia.

---

# 32. Ablation Minimal

Karena penelitian sengaja disederhanakan, ablation juga jangan banyak.

Cukup **satu ablation utama**:

$$
\boxed{
SNN
\quad vs\quad
SNN+N\text{-}gram
}
$$

dan kalau masih ada ruang:

$$
N=1
$$

vs.

$$
N=1,2,3,4
$$

Tetapi eksperimen kedua ini cukup sebagai analisis tambahan, **bukan RQ baru**.

---

# 33. Expected Result Structure

Tabel utama:

| Model        | Params | BLEU | chrF | Tok/s | Spike/Token |
| ------------ | -----: | ---: | ---: | ----: | ----------: |
| SNN          |    ~1M |    — |    — |     — |           — |
| SNN + N-gram |    ~1M |    — |    — |     — |           — |

Kemudian tabel kedua:

| Model        | RAM | Latency | Spike Rate |
| ------------ | --: | ------: | ---------: |
| SNN          |   — |       — |          — |
| SNN + N-gram |   — |       — |          — |

---

# 34. Statistical Reliability

Karena SNN training dapat memiliki stochasticity, jangan hanya menjalankan sekali.

Gunakan:

$$
seed\in\{1,2,3\}
$$

Minimal tiga independent runs untuk eksperimen utama.

Laporan:

$$
BLEU=
\mu\pm\sigma
$$

Contoh:

$$
BLEU=18.42\pm0.31
$$

Kemudian perbedaan:

$$
\Delta BLEU
$$

dapat dilaporkan bersama variasinya.

---

# 35. Research Flow

Akhirnya seluruh penelitian hanya memiliki alur:

```text
                RESEARCH QUESTION
                       │
                       ▼
       Can a ~1M Fully-Spiking Seq2Seq
         perform machine translation?
                       │
                       ▼
                Build SNN Baseline
                       │
                       ▼
             Evaluate Translation
                       │
                       ▼
              Add N-gram Memory
                       │
                       ▼
           Compare SNN vs SNN+NG
                       │
                       ▼
         Measure Quality + Efficiency
                       │
                       ▼
                  CONCLUSION
```

---

# 36. Posisi Kontribusi Penelitian

Dengan metodologi ini, **jangan mengklaim tiga novelty berbeda**.

Kontribusi utamanya cukup satu:

$$
\boxed{
\text{Sparse N-gram Memory-Augmented Fully Spiking Seq2Seq}
}
$$

dengan dua aspek evaluasi:

$$
\text{Translation capability}
$$

dan:

$$
\text{Computational efficiency}
$$

Delta-BPTT adalah **mekanisme training**.

Spike-based output adalah **mekanisme inference**.

N-gram memory adalah **komponen kontribusi arsitektur utama**.

Dengan pembagian ini, paper menjadi jauh lebih fokus:

> **Small SNN → apakah bisa translation?**

lalu:

> **Tambahkan external N-gram memory → apakah menjadi lebih baik?**

lalu:

> **Seberapa mahal computationally dibandingkan baseline?**

Itu sudah cukup untuk membentuk satu cerita penelitian yang utuh tanpa memaksa kita membuktikan enam klaim ilmiah berbeda.
