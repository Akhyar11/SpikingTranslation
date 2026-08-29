import jax
import jax.numpy as jnp
import numpy as np
from dataset import StreamingCorpus, SparseNGramMemory
from snn_core import (
    encoder_step, stcm_encoder_step, stcm_decoder_step, decoder_step,
    precompute_all_sdr
)

# Hyperparameters (must match train.py)
K = 20
beta_seq = 0.5
d_e = 64
d_c = 128
d_d = 128
num_active_sdr = 3

def infer(params, src_batch, max_len, m_v_all, ngram_mem=None):
    # src_batch: (B, T_src)
    B, T_src = src_batch.shape
    vocab_tgt = params['dec_w_y'].shape[1]
    
    # Init states
    u_e = jnp.zeros((B, d_e))
    s_e_prev = jnp.zeros((B, K, d_e))
    u_c_enc = jnp.zeros((B, d_c))
    s_c_enc_prev = jnp.zeros((B, K, d_c))
    
    # --- ENCODER PHASE ---
    # Not using lax.scan here just to show manual unrolling, but scan is better
    # We will use scan for simplicity
    def encode_scan(carry, x_t):
        u_e, s_e_prev, u_c, s_c_prev = carry
        s_x = jax.nn.one_hot(x_t, params['enc_w_e'].shape[1])
        s_e_new, s_c_new = [], []
        for tau in range(K):
            u_e, s_e_tau = encoder_step(u_e, s_e_prev[:, tau], s_x, params['enc_w_e'], params['enc_w_r'], beta_seq)
            u_c, s_c_tau = stcm_encoder_step(u_c, s_c_prev[:, tau], s_e_tau, params['stcm_w_ce'], params['stcm_w_cc'], beta_seq)
            s_e_new.append(s_e_tau)
            s_c_new.append(s_c_tau)
        return (u_e, jnp.stack(s_e_new, axis=1), u_c, jnp.stack(s_c_new, axis=1)), None

    src_batch_T = jnp.swapaxes(src_batch, 0, 1)
    (u_e, s_e_prev, u_c_ctx, s_c_ctx), _ = jax.lax.scan(encode_scan, (u_e, s_e_prev, u_c_enc, s_c_enc_prev), src_batch_T)
    
    # --- DECODER PHASE (Autoregressive) ---
    u_d = jnp.zeros((B, d_d))
    s_d_prev = jnp.zeros((B, K, d_d))
    
    current_tokens = jnp.full((B,), 2, dtype=jnp.int32) # <BOS>
    results = []
    
    for t in range(max_len):
        s_y = jax.nn.one_hot(current_tokens, vocab_tgt)
        s_c_new, s_d_new = [], []
        
        for tau in range(K):
            u_c_ctx, s_c_tau = stcm_decoder_step(u_c_ctx, s_c_ctx[:, tau], s_d_prev[:, tau], params['stcm_w_ctx'], params['stcm_w_self'], beta_seq)
            u_d, s_d_tau = decoder_step(u_d, s_d_prev[:, tau], s_y, s_c_tau, params['dec_w_y'], params['dec_w_c'], params['dec_w_r'], beta_seq)
            s_c_new.append(s_c_tau)
            s_d_new.append(s_d_tau)
            
        s_c_ctx = jnp.stack(s_c_new, axis=1)
        s_d_prev = jnp.stack(s_d_new, axis=1)
        
        # Calculate SDR Scores
        s_d_sum = jnp.sum(s_d_prev, axis=1) # (B, d_d)
        
        def calc_scores(batch_s_d_sum):
            return jnp.sum(batch_s_d_sum[m_v_all], axis=1)
            
        scores = jax.vmap(calc_scores)(s_d_sum) # (B, vocab_tgt)
        scores = np.array(scores) # Bring to CPU for NGram loop
        
        # NGram Integration (Done on CPU because dict lookup is not jittable easily)
        for b in range(B):
            prev_tok = int(current_tokens[b])
            if ngram_mem:
                cands = ngram_mem.get_candidates(prev_tok)
                for (tok, prob) in cands:
                    if tok < vocab_tgt:
                        scores[b, tok] += prob * 2.0
            
            # Mask out PAD and UNK
            scores[b, 0] = -np.inf
            scores[b, 1] = -np.inf
            
        next_tokens = jnp.argmax(scores, axis=1)
        results.append(next_tokens)
        current_tokens = next_tokens
        
    return jnp.stack(results, axis=1) # (B, max_len)

def main():
    print("Memuat Vocabulary dan N-Gram Memory...")
    corpus = StreamingCorpus("../dataset/OpenSubtitles.en-id.en", "../dataset/OpenSubtitles.en-id.id")
    vocab_tgt = corpus.vocab_size()
    
    ngram = SparseNGramMemory()
    ngram.build_from_corpus(corpus, 500000)
    
    print("Memuat Checkpoint Model...")
    try:
        ckpt = np.load("best_model_jax.npz")
        params = {k: jnp.array(v) for k, v in ckpt.items()}
    except:
        print("Checkpoint best_model_jax.npz tidak ditemukan. Anda harus training dulu.")
        return
        
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    texts = [
        "-=Episode 13=-",
        "People.",
        "That brat."
    ]
    
    for text in texts:
        ids = corpus.encode(text)
        src_batch = jnp.array([ids])
        out_ids = infer(params, src_batch, max_len=20, m_v_all=m_v_all, ngram_mem=ngram)
        out_text = corpus.decode(out_ids[0].tolist())
        print("En:", text)
        print("Id:", out_text.replace("<EOS>", "").strip())
        print("------------------")

if __name__ == "__main__":
    main()
