Bisa. Dan kalau tujuannya supaya kamu **benar-benar bisa melihat mekanisme model dari forward → loss → backward → update parameter**, saya sarankan kita tuliskan matematikanya sampai level indeks waktu dan indeks neuron, bukan hanya persamaan blok.

Ada satu koreksi penting dari jawaban saya sebelumnya: **jangan langsung menganggap persamaan reset-aware sebagai “Delta-BPTT milik kita” tanpa membedakan dengan paper Yang (2020)**. Paper tersebut memang secara khusus menambahkan dependensi temporal akibat reset, dan justru melaporkan bahwa manfaat tambahan itu tidak selalu signifikan pada task besar. Jadi ini harus menjadi **bagian eksperimen/challenge**, bukan asumsi bahwa selalu lebih baik. ([arXiv][1])

---

# 0. Baseline Arsitektur Eksperimen Terkunci (STCM & Kontinuitas Membran)

*(Catatan: Bagian ini merupakan rancangan matematis final yang telah dikunci dan menggantikan beberapa eksplorasi awal di bagian-bagian bawah dokumen ini.)*

Arsitektur ini didesain secara presisi untuk menyelesaikan 5 tantangan utama SNN dalam NLP dengan solusi independen yang bersih:

| Masalah | Solusi |
|---|---|
| $T_x \neq T_y$ | **Spiking Temporal Context Memory (STCM)** |
| Sub-threshold information loss | **Inter-token membrane carry** |
| Reset gradient terputus | **Reset-aware Delta-BPTT** |
| Vocabulary layer yang masif | **Subword Tokenization (BPE) & Sparse N-gram candidate memory** |
| Softmax / dense output | **Spike accumulation + argmax margin loss** |

*The proposed architecture maintains membrane-state continuity across both intra-token neural dynamics and inter-token sequence dynamics, while a dedicated Spiking Temporal Context Memory bridges the variable-length source and target sequences.*

---

## 0.1 Forward Pass Lengkap

### Prinsip Kontinuitas Membran Lintas Token (Inter-token Boundary)
Untuk **semua** lapisan (Encoder, Decoder, dan STCM), sisa voltase membran dari akhir token sebelumnya diwariskan ke awal token saat ini:
$$ U_{t}^{0} = \beta_{\text{seq}} U_{t-1}^{K} $$

### A. Spiking Encoder (Source)
$$ I_{e,t}^{\tau} = W_e S_{x_t}^{\tau} + W_r S_{e,t-1}^{\tau} $$
$$ U_{e,t}^{\tau} = \beta_e U_{e,t}^{\tau-1} + I_{e,t}^{\tau} - \vartheta_e S_{e,t}^{\tau-1} $$
$$ S_{e,t}^{\tau} = H(U_{e,t}^{\tau} - \vartheta_e) $$

### B. STCM (Source-side Context Building)
$$ I_{c,t}^{\tau} = W_{ce} S_{e,t}^{\tau} + W_{cc} S_{c,t-1}^{\tau} $$
$$ U_{c,t}^{\tau} = \beta_c U_{c,t}^{\tau-1} + I_{c,t}^{\tau} - \vartheta_c S_{c,t}^{\tau-1} $$
$$ S_{c,t}^{\tau} = H(U_{c,t}^{\tau} - \vartheta_c) $$
**State Konteks Akhir (T_x):**
$$ S_{ctx,0}^{\tau} = S_{c,T_x}^{\tau} $$

### C. STCM (Decoder-side Dynamic Context)
$$ I_{ctx,t}^{\tau} = W_{ctx} S_{d,t-1}^{\tau} + W_{self} S_{ctx,t-1}^{\tau} $$
$$ U_{ctx,t}^{\tau} = \beta_{ctx} U_{ctx,t}^{\tau-1} + I_{ctx,t}^{\tau} - \vartheta_{ctx} S_{ctx,t}^{\tau-1} $$
$$ S_{ctx,t}^{\tau} = H(U_{ctx,t}^{\tau} - \vartheta_{ctx}) $$

### D. Spiking Decoder (Target)
$$ I_{d,t}^{\tau} = W_y S_{y_{t-1}}^{\tau} + W_c S_{ctx,t}^{\tau} + W_r S_{d,t-1}^{\tau} $$
$$ U_{d,t}^{\tau} = \beta_d U_{d,t}^{\tau-1} + I_{d,t}^{\tau} - \vartheta_d S_{d,t}^{\tau-1} $$
$$ S_{d,t}^{\tau} = H(U_{d,t}^{\tau} - \vartheta_d) $$

---

## 0.2 Spatiotemporal Delta-BPTT Backward Pass

Gradien melangkah mundur dengan dua fase waktu yang bergantian: **Intra-token (waktu neural)** dan **Inter-token Boundary (waktu kata)**.

### A. Intra-token Neural Dynamics ($\tau \to \tau-1$)
$$ \delta_t^{\tau-1} = \frac{\partial L}{\partial S_t^{\tau-1}} g_t^{\tau-1} + \delta_t^{\tau} (\beta - \vartheta g_t^{\tau-1}) $$

### B. Inter-token Boundary Continuity ($0 \to K$ dari kata sebelumnya)
$$ \delta_{t-1}^{K} \leftarrow \delta_{t-1}^{K} + \beta_{\text{seq}} \delta_t^0 $$

### C. Alur Propagasi Eksternal
1. **Loss $\to$ Decoder:**
   $$ \frac{\partial L}{\partial S_{d,t}^\tau} = \text{Margin Gradient} + W_r^T \delta_{d,t+1}^\tau + W_{ctx}^T \delta_{ctx, t+1}^\tau $$
2. **Decoder $\to$ STCM (Decoder-side):**
   $$ \frac{\partial L}{\partial S_{ctx,t}^\tau} = W_c^T \delta_{d,t}^\tau + W_{self}^T \delta_{ctx, t+1}^\tau $$
3. **STCM (Decoder-side $t=0$) $\to$ STCM (Source-side $t=T_x$):**
   $$ \frac{\partial L}{\partial S_{c,T_x}^\tau} = \frac{\partial L}{\partial S_{ctx,0}^\tau} $$
4. **STCM (Source-side):**
   $$ \frac{\partial L}{\partial S_{c,t}^\tau} = W_{cc}^T \delta_{c,t+1}^\tau $$
5. **STCM $\to$ Encoder:**
   $$ \frac{\partial L}{\partial S_{e,t}^\tau} = W_{ce}^T \delta_{c,t}^\tau + W_r^T \delta_{e,t+1}^\tau $$

---

# 1. Struktur matematis keseluruhan (Eksplorasi Awal)

Kita definisikan model:

$$
\boxed{
X
\rightarrow
\text{N-gram Memory}
\rightarrow
\text{Spiking Encoder}
\rightarrow
\text{Spiking State}
\rightarrow
\text{Spiking Decoder}
\rightarrow
\text{Sparse Vocabulary Memory}
\rightarrow
\text{Spike Selection}
}
$$

Ada **dua dimensi waktu** yang sangat penting:

### Sequence time

$$
t=1,\ldots,T
$$

Ini adalah posisi token.

### Neural time

$$
\tau=1,\ldots,K
$$

Ini adalah timestep internal SNN untuk memproses satu token.

Jadi satu token:

$$
x_t
$$

tidak langsung menjadi satu aktivasi.

Melainkan:

$$
x_t
\rightarrow
S_{x_t}^{1},
S_{x_t}^{2},
\ldots,
S_{x_t}^{K}
$$

---

# 2. N-gram Memory

Misalkan source:

$$
X=(x_1,x_2,\ldots,x_{T_x})
$$

Untuk N-gram:

$$
g_t^{(n)}
=
(x_{t-n+1},\ldots,x_t)
$$

misalnya trigram:

$$
g_t^{(3)}
=
(x_{t-2},x_{t-1},x_t)
$$

Memory:

$$
M_n(g_t^{(n)})
$$

mengembalikan sparse temporal spike pattern.

Kita definisikan:

$$
S_{\mathrm{mem},t}^{(n),\tau}
\in
\{0,1\}^{D}
$$

Sehingga gabungan memory:

$$
I_{\mathrm{mem},t}^{\tau}
=
\sum_{n\in\mathcal N}
\alpha_n
W_n
S_{\mathrm{mem},t}^{(n),\tau}
$$

dengan:

$$
\mathcal N=\{1,2,3,4\}
$$

---

# 3. Tetapi ada masalah penting

Kalau:

$$
S_{\mathrm{mem}}
$$

hanya hasil lookup N-gram, maka **memory bukan sesuatu yang dilatih melalui BPTT**.

Ini justru bagus untuk konsep yang kamu inginkan.

Kita punya:

$$
\boxed{
\theta_{\mathrm{SNN}}
\quad\text{trainable}
}
$$

sedangkan:

$$
\boxed{
M_{\mathrm{ngram}}
\quad\text{non-parametric memory}
}
$$

Jadi:

$$
P_{\mathrm{trainable}}\approx1M
$$

sementara:

$$
|M_{\mathrm{ngram}}|
\gg
P_{\mathrm{trainable}}
$$

Tetapi hanya sebagian memory yang aktif:

$$
M_t\subset M
$$

---

# 4. Forward LIF

Untuk neuron ke-\(i\):

$$
U_{i,t}^{\tau}
=
\beta U_{i,t}^{\tau-1}
+
I_{i,t}^{\tau}
-
\vartheta S_{i,t}^{\tau-1}
$$

Kemudian spike:

$$
\boxed{
S_{i,t}^{\tau}
=
H(U_{i,t}^{\tau}-\vartheta)
}
$$

dengan:

$$
S_{i,t}^{\tau}\in\{0,1\}
$$

Ini adalah forward computation yang sebenarnya.

Tidak ada:

$$
ReLU
$$

Tidak ada:

$$
GELU
$$

Tidak ada:

$$
Softmax
$$

di dalam neuron.

---

# 5. Input Current

Untuk encoder:

$$
I_{e,t}^{\tau}
=
W_eS_{x_t}^{\tau}
+
W_rS_{e,t-1}^{\tau}
+
I_{\mathrm{mem},t}^{\tau}
$$

sehingga:

$$
U_{e,t}^{\tau}
=
\beta_eU_{e,t}^{\tau-1}
+
W_eS_{x_t}^{\tau}
+
W_rS_{e,t-1}^{\tau}
+
I_{\mathrm{mem},t}^{\tau}
-
\vartheta_eS_{e,t}^{\tau-1}
$$

dan:

$$
S_{e,t}^{\tau}
=
H(U_{e,t}^{\tau}-\vartheta_e)
$$

---

# 6. Decoder

Decoder menerima:

1. target token sebelumnya,
2. encoder state,
3. recurrent state.

Forward:

$$
I_{d,t}^{\tau}
=
W_dS_{y_{t-1}}^{\tau}
+
W_cH_{e,t}^{\tau}
+
W_rS_{d,t-1}^{\tau}
$$

Kemudian:

$$
U_{d,t}^{\tau}
=
\beta_dU_{d,t}^{\tau-1}
+
I_{d,t}^{\tau}
-
\vartheta_dS_{d,t}^{\tau-1}
$$

dan:

$$
\boxed{
S_{d,t}^{\tau}
=
H(U_{d,t}^{\tau}-\vartheta_d)
}
$$

---

# 7. Temporal state

Kita tidak ingin mengubah semua spike menjadi satu vector ANN.

Sebaliknya:

$$
H_{e,t}
=
\{S_{e,t}^{1},\ldots,S_{e,t}^{K}\}
$$

Jadi context tetap temporal:

$$
\boxed{
H_e^{1:K}
}
$$

Decoder menerima:

$$
H_e^\tau
$$

pada setiap neural timestep.

---

# 8. Output tanpa Softmax

Ini bagian yang menurut saya paling penting dari desainmu.

Jangan:

$$
S_D
\rightarrow
W_{\mathrm{out}}
\rightarrow
z
\rightarrow
Softmax(z)
$$

Kita gunakan sparse vocabulary memory.

Untuk token kandidat \(v\):

$$
M_v
=
\{j_1,j_2,\ldots,j_r\}
$$

Kemudian:

$$
A_t(v)
=
\sum_{\tau=1}^{K}
\sum_{j\in M_v}
S_{d,t,j}^{\tau}
$$

Jadi token mendapatkan "suara" berdasarkan spike.

Kemudian:

$$
\boxed{
\hat y_t
=
\arg\max_{v\in C_t} A_t(v)
}
$$

Ini benar-benar spike-based decoding.

---

# 9. Candidate vocabulary

Kita tidak melakukan:

$$
V=32,000
$$

atau bahkan:

$$
V=100,000
$$

full output setiap timestep.

Kita buat:

$$
C_t\subset V
$$

dan:

$$
|C_t|\ll V
$$

Misalnya:

$$
|C_t|=256
$$

Candidate berasal dari:

$$
C_t
=
C_t^{ngram}
\cup
C_t^{copy}
\cup
C_t^{freq}
\cup
C_t^{special}
$$

---

# 10. Loss

Karena kita tidak memakai probabilitas Softmax, saya lebih menyukai **margin spike loss**.

Ground truth:

$$
y_t
$$

Score benar:

$$
A_t^+
=
A_t(y_t)
$$

Negative terbaik:

$$
A_t^-
=
\max_{v\in C_t,v\neq y_t}
A_t(v)
$$

Loss:

$$
\boxed{
L_t
=
\max
\left(
0,
m-A_t^++A_t^-
\right)
}
$$

Total:

$$
\boxed{
L=
\sum_{t=1}^{T_y}L_t
}
$$

---

# 11. Backward dimulai dari spike score

Karena:

$$
A_t(v)
=
\sum_{\tau}
\sum_j
S_{d,t,j}^{\tau}
$$

maka:

$$
\frac{\partial A_t(v)}
{\partial S_{d,t,j}^{\tau}}
=
\begin{cases}
1,&j\in M_v\\
0,&j\notin M_v
\end{cases}
$$

Untuk margin loss ketika aktif:

$$
m-A_t^++A_t^->0
$$

maka:

$$
\frac{\partial L_t}
{\partial A_t^+}
=
-1
$$

dan:

$$
\frac{\partial L_t}
{\partial A_t^-}
=
1
$$

Dengan demikian:

$$
\boxed{
\frac{\partial L_t}
{\partial S_{d,t,j}^{\tau}}
}
$$

menjadi sumber gradient pertama menuju SNN.

---

# 12. Masalah Heaviside

Forward:

$$
S_{d,t,j}^{\tau}
=
H(U_{d,t,j}^{\tau}-\vartheta)
$$

Secara matematis:

$$
\frac{\partial H(x)}{\partial x}
$$

tidak dapat digunakan secara normal.

Maka:

$$
\boxed{
\frac{\partial S}
{\partial U}
\approx
g(U-\vartheta)
}
$$

misalnya Rectangular Surrogate Mask (Boxcar Gradient):

$$
g(z)
=
\begin{cases}
1, & |z| < w \\
0, & \text{lainnya}
\end{cases}
$$

di mana $w$ adalah surrogate window width (misal $w=1.0$).

---

# 13. Backward dasar

Tanpa memperhatikan reset terlebih dahulu:

$$
\frac{\partial L}
{\partial U_t^\tau}
=
\frac{\partial L}
{\partial S_t^\tau}
g(U_t^\tau-\vartheta)
+
\frac{\partial L}
{\partial U_t^{\tau+1}}
\frac{\partial U_t^{\tau+1}}
{\partial U_t^\tau}
$$

Karena:

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
\vartheta
\frac{\partial S_t^\tau}
{\partial U_t^\tau}
$$

sehingga:

$$
\boxed{
\delta_t^\tau
=
\frac{\partial L}{\partial S_t^\tau}g_t^\tau
+
\delta_t^{\tau+1}
\left(
\beta-\vartheta g_t^\tau
\right)
}
$$

dengan:

$$
g_t^\tau
=
g(U_t^\tau-\vartheta)
$$

atau:

$$
\boxed{
\delta_t^\tau
=
\frac{\partial L}{\partial S_t^\tau}g_t^\tau
+
\beta\delta_t^{\tau+1}
-
\vartheta g_t^\tau\delta_t^{\tau+1}
}
$$

---

# 14. Inilah reset-aware Delta-BPTT

Perhatikan ada dua jalur:

### Direct spike path

$$
\frac{\partial L}{\partial S_t^\tau}
g_t^\tau
$$

dan:

### Temporal membrane path

$$
\beta\delta_t^{\tau+1}
$$

serta:

### Reset path

$$
-\vartheta g_t^\tau\delta_t^{\tau+1}
$$

Sehingga:

$$
\boxed{
\underbrace{
\delta_t^\tau
}_{\text{total gradient}}
=
\underbrace{
\frac{\partial L}{\partial S_t^\tau}g_t^\tau
}_{\text{spike}}
+
\underbrace{
\beta\delta_t^{\tau+1}
}_{\text{membrane}}
-
\underbrace{
\vartheta g_t^\tau\delta_t^{\tau+1}
}_{\text{reset}}
}
$$

**Ini adalah mekanisme yang seharusnya benar-benar kamu eksperimenkan.**

Paper Yang secara eksplisit membahas tambahan temporal dependency dari reset seperti ini. Mereka menemukan manfaatnya pada toy task, tetapi tidak selalu pada task yang lebih besar. ([arXiv][1])

Jadi justru di Seq2Seq translation kamu punya pertanyaan penelitian yang menarik:

> Apakah reset-aware temporal gradient menjadi lebih penting ketika SNN harus mempertahankan informasi temporal untuk menghasilkan sequence, dibandingkan classification?

---

# 15. Gradient terhadap input current

Karena:

$$
U_t^\tau
=
\beta U_t^{\tau-1}
+
I_t^\tau
-
\vartheta S_t^{\tau-1}
$$

maka:

$$
\boxed{
\frac{\partial L}
{\partial I_t^\tau}
=
\delta_t^\tau
}
$$

---

# 16. Gradient terhadap synaptic weight

Misalnya:

$$
I_t^\tau
=
W S_t^\tau
$$

maka:

$$
\frac{\partial I_t^\tau}
{\partial W}
=
S_t^\tau
$$

sehingga:

$$
\boxed{
\frac{\partial L}{\partial W}
=
\sum_{t}
\sum_{\tau}
\delta_t^\tau
(S_t^\tau)^T
}
$$

Untuk neuron spesifik:

$$
\boxed{
\frac{\partial L}
{\partial W_{ij}}
=
\sum_{t,\tau}
\delta_{i,t}^{\tau}
S_{j,t}^{\tau}
}
$$

Ini adalah weight update yang sebenarnya.

---

# 17. Update parameter

Dengan learning rate:

$$
\eta
$$

maka SGD:

$$
\boxed{
W
\leftarrow
W-\eta\frac{\partial L}{\partial W}
}
$$

Kalau Adam digunakan:

$$
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

$$
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

kemudian:

$$
W
\leftarrow
W
-
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

Tetapi untuk paper, **matematika SNN sebaiknya tetap ditulis sampai gradient**, sementara optimizer dilaporkan sebagai training detail.

---

# 18. Backward melalui recurrent connection

Sekarang masuk bagian yang lebih penting untuk Seq2Seq.

Misalnya:

$$
I_{t}^{\tau}
=
W_rS_{t-1}^{\tau}
+
W_xS_{x_t}^{\tau}
+
I_{\mathrm{mem},t}^{\tau}
$$

Gradient terhadap recurrent spike:

$$
\frac{\partial L}
{\partial S_{t-1}^{\tau}}
=
W_r^T
\delta_t^\tau
+
\text{gradient dari timestep lain}
$$

Karena:

$$
S_{t-1}^{\tau}
=
H(U_{t-1}^{\tau}-\vartheta)
$$

maka:

$$
\boxed{
\frac{\partial L}
{\partial U_{t-1}^{\tau}}
=
\frac{\partial L}
{\partial S_{t-1}^{\tau}}
g_{t-1}^{\tau}
+
\text{temporal gradient}
}
$$

Artinya gradient bergerak dalam **dua arah**:

$$
\boxed{
(t,\tau)
\rightarrow
(t,\tau-1)
}
$$

dan:

$$
\boxed{
(t,\tau)
\rightarrow
(t-1,\tau)
}
$$

Ini sangat penting.

---

# 19. Jadi computational graph sebenarnya

Bukan:

```text
Token
  ↓
SNN
  ↓
Output
```

Tetapi:

```text
                  neural time τ
             1      2      3     ... K
             ↓      ↓      ↓       ↓
token t →   LIF →  LIF →  LIF → ... LIF
             ↑      ↑      ↑
             │      │      │
             └──────┴──────┴──── temporal recurrence
                    ↑
                    │
             token-time recurrence
                    │
token t-1 ──────────┘
```

Jadi ada **spatiotemporal BPTT**:

$$
\boxed{
\text{token-time BPTT}
+
\text{neural-time BPTT}
}
$$

---

# 20. Encoder → Decoder gradient

Decoder bergantung pada encoder:

$$
U_{d,t}^{\tau}
\supset
W_cS_{e,T_x}^{\tau}
$$

Maka:

$$
\frac{\partial L}
{\partial S_{e,T_x}^{\tau}}
\mathrel{+}=
W_c^T
\delta_{d,t}^{\tau}
$$

untuk semua target timestep:

$$
\boxed{
\frac{\partial L}
{\partial S_{e,T_x}^{\tau}}
=
\sum_{t=1}^{T_y}
W_c^T
\delta_{d,t}^{\tau}
}
$$

Kemudian gradient tersebut masuk kembali ke encoder melalui Delta-BPTT.

---

# 21. Full encoder gradient

Untuk setiap encoder timestep:

$$
\delta_{e,t}^{\tau}
=
\frac{\partial L}
{\partial S_{e,t}^{\tau}}
g_{e,t}^{\tau}
+
\beta_e
\delta_{e,t}^{\tau+1}
-
\vartheta_e
g_{e,t}^{\tau}
\delta_{e,t}^{\tau+1}
$$

ditambah gradient recurrent dari:

$$
S_{e,t}^{\tau}
\rightarrow
S_{e,t+1}^{\tau}
$$

Jadi secara konseptual:

$$
\boxed{
\delta_{e,t}^{\tau}
=
\text{local spike gradient}
+
\text{neural-time gradient}
+
\text{token-time gradient}
}
$$

---

# 22. Gradient ke N-gram memory

Ini harus dibedakan.

Jika memory:

$$
M(g)
$$

adalah **fixed lookup**, maka:

$$
\boxed{
\frac{\partial L}{\partial M(g)}=0
}
$$

dalam training SNN.

Namun gradient terhadap memory injection:

$$
\frac{\partial L}
{\partial I_{\mathrm{mem},t}^{\tau}}
=
\delta_{e,t}^{\tau}
$$

tetap tersedia.

Jadi kita bisa mengukur:

$$
\left\|
\frac{\partial L}
{\partial I_{\mathrm{mem}}}
\right\|
$$

untuk mengetahui apakah memory benar-benar membantu neural computation.

---

# 23. Kalau memory ingin dilatih

Baru kita definisikan:

$$
S_{\mathrm{mem}}
=
f_\phi(g)
$$

dengan parameter:

$$
\phi
$$

dan:

$$
\frac{\partial L}{\partial\phi}
=
\sum_{t,\tau}
\frac{\partial L}{\partial I_{\mathrm{mem},t}^{\tau}}
\frac{\partial I_{\mathrm{mem},t}^{\tau}}
{\partial S_{\mathrm{mem},t}^{\tau}}
\frac{\partial S_{\mathrm{mem},t}^{\tau}}
{\partial\phi}
$$

Tetapi **untuk versi 1 saya tidak menyarankan ini**.

Karena tujuanmu justru:

$$
\boxed{
\text{small trainable SNN}
+
\text{large non-parametric memory}
}
$$

---

# 24. Forward lengkap

Secara keseluruhan:

### Step 1

$$
g_t^{(1)},g_t^{(2)},g_t^{(3)},g_t^{(4)}
$$

### Step 2

$$
M(g_t)
\rightarrow
S_{\mathrm{mem},t}^{1:K}
$$

### Step 3

Encoder:

$$
S_x
\rightarrow
U_e
\rightarrow
S_e
$$

### Step 4

Context:

$$
S_e^{1:K}
$$

### Step 5

Decoder:

$$
S_{y_{t-1}}
+
S_e
\rightarrow
U_d
\rightarrow
S_d
$$

### Step 6

Output spike accumulation:

$$
S_d
\rightarrow
A_t(v)
$$

### Step 7

Prediction:

$$
\hat y_t
=
\arg\max_vA_t(v)
$$

---

# 25. Backward lengkap

Urutannya:

$$
L
$$

↓

$$
\frac{\partial L}{\partial A_t}
$$

↓

$$
\frac{\partial L}{\partial S_d^\tau}
$$

↓

$$
\frac{\partial L}{\partial U_d^\tau}
$$

↓

**Delta-BPTT**

$$
\delta_d^\tau
=
\text{local}
+
\text{membrane}
+
\text{reset}
$$

↓

$$
\frac{\partial L}{\partial W_d},
\frac{\partial L}{\partial W_r},
\frac{\partial L}{\partial W_c}
$$

↓

Encoder:

$$
\frac{\partial L}{\partial S_e^\tau}
$$

↓

Encoder Delta-BPTT:

$$
\delta_e^\tau
$$

↓

$$
\frac{\partial L}{\partial W_e}
$$

↓

update:

$$
W\leftarrow W-\eta\nabla W
$$

---

# 26. Hal yang menurut saya justru menjadi novelty

Kalau kita formulasi seperti ini, novelty-nya bukan:

> "Saya membuat SNN untuk translation."

Itu terlalu lemah.

Novelty yang jauh lebih menarik:

$$
\boxed{
\text{Memory-Augmented Fully Spiking Seq2Seq}
}
$$

dengan:

$$
\boxed{
P_{\mathrm{trainable}}\approx1M
}
$$

tetapi:

$$
P_{\mathrm{memory}}\gg P_{\mathrm{trainable}}
$$

dan:

$$
\boxed{
\text{temporal spike computation}
}
$$

tetap dilakukan pada:

$$
t\times\tau
$$

bukan mengganti SNN dengan ANN kecil.

---

# 27. Paper yang paling tepat untuk di-challenge

Saya akan ubah prioritasnya menjadi **empat**, bukan banyak-banyak paper.

### Challenge 1 — Qwen/Engram

Ini adalah **architectural challenge**:

$$
\text{large sparse N-gram memory}
+
\text{small active computation}
$$

Proposalmu:

$$
\text{large sparse N-gram memory}
+
\textbf{small active SNN computation}
$$

Jadi pertanyaan:

> Apakah active computation yang sama-sama kecil dapat digantikan oleh temporal SNN dan tetap menghasilkan translation yang kompetitif?

Ini merupakan challenge konseptual, bukan klaim bahwa implementasinya identik dengan Qwen.

---

### Challenge 2 — Yang 2020

**Temporal Surrogate Back-propagation for Spiking Neural Networks.**

Ini adalah challenge paling penting untuk matematika Delta-BPTT.

Paper tersebut secara eksplisit memasukkan dependency akibat reset ke temporal gradient, tetapi menemukan keuntungan yang lebih jelas pada toy problem daripada CIFAR-10. ([arXiv][1])

Maka eksperimenmu:

$$
\boxed{
\text{Reset-aware}
\quad vs\quad
\text{Reset-detached}
}
$$

pada **machine translation**.

Itu jauh lebih menarik daripada sekadar memakai rumus paper tersebut.

---

### Challenge 3 — STAR-SNN

**STAR-SNN**, *Neurocomputing*, 674, 132968 (2026).

Paper ini sangat relevan karena fokusnya pada recurrent SNN, temporal learning, surrogate gradient, BPTT, dan hardware efficiency. Mereka menggunakan TBPTT dengan \(K=1\) untuk menekan biaya memori, tetapi konsekuensinya adalah gradient propagation sepanjang sequence menjadi terbatas. ([ScienceDirect][2])

Di sini proposalmu bisa mengambil posisi berbeda:

$$
\boxed{
K>1
+
\text{full temporal gradient}
}
$$

dan menguji apakah memory eksternal memungkinkan kita mempertahankan model kecil tanpa harus memotong temporal learning terlalu agresif.

---

### Challenge 4 — SNN BPTT efficiency

Di sini yang kita challenge bukan satu persamaan tertentu, tetapi paradigma:

$$
\text{BPTT}
\rightarrow
\text{large temporal memory}
$$

Proposalmu:

$$
\text{small trainable core}
+
\text{sparse external memory}
$$

Kemudian ukur:

$$
\text{BLEU}
$$

vs.

$$
\text{trainable parameters}
$$

vs.

$$
\text{spikes/token}
$$

vs.

$$
\text{latency/token}
$$

vs.

$$
\text{RAM}
$$

---

# 28. Satu hal yang harus kita ubah dari desain sebelumnya

Saya **tidak menyarankan langsung memakai final encoder state saja**:

$$
H_{enc}=S_{e,T_x}
$$

untuk translation serius.

Itu membuat bottleneck:

$$
x_1,\ldots,x_{T_x}
\rightarrow
H_{enc}
$$

yang berpotensi kehilangan informasi.

Untuk eksperimen toy boleh.

Untuk paper utama, kita sebaiknya mengembangkan **spiking temporal cross-context mechanism** sehingga decoder bisa mengakses representasi temporal encoder:

$$
\boxed{
S_{e,1:T_x}^{1:K}
}
$$

bukan hanya:

$$
S_{e,T_x}^{1:K}
$$

Dan ini bisa kita desain **tanpa mengubahnya menjadi Transformer ANN biasa**.

Itu justru bagian yang menurut saya paling layak kita bedah berikutnya, karena kalau target akhirnya **machine translation**, bottleneck encoder-decoder ini bisa menjadi titik lemah terbesar arsitektur 1M parametermu.

Dengan formulasi di atas, mekanisme matematis utamanya sudah lengkap:

$$
\boxed{
\text{N-gram}
\rightarrow
\text{Spike Encoding}
\rightarrow
\text{LIF Forward}
\rightarrow
\text{Spike Output}
\rightarrow
\text{Margin Loss}
\rightarrow
\text{Surrogate Gradient}
\rightarrow
\text{Reset-aware Delta-BPTT}
\rightarrow
\text{Temporal + Token-time Backprop}
\rightarrow
\text{Weight Update}
}
$$

Dan yang paling penting: **Softmax memang sudah keluar dari jalur utama model**.

[1]: https://arxiv.org/abs/2011.09964?utm_source=chatgpt.com "Temporal Surrogate Back-propagation for Spiking Neural Networks"
[2]: https://www.sciencedirect.com/science/article/pii/S0925231226003656?utm_source=chatgpt.com "STAR-SNN: A spatio-temporal adaptive recurrent spiking neural network with separated propagation surrogate gradient for hardware efficient real-time learning - ScienceDirect"
