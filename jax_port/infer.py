import jax
import jax.numpy as jnp
import numpy as np
import functools
from dataset import StreamingCorpus, SparseNGramMemory
from snn_core import (
    encoder_step, stcm_encoder_step, stcm_decoder_step, decoder_step,
    precompute_all_sdr
)

# Hyperparameters (must match train.py)
K = 5
beta_seq = 0.1
d_e = 64
d_c = 128
d_d = 128
num_active_sdr = 3

@functools.partial(jax.jit, static_argnums=(2,))
def encode_batch(params, src_batch, K_len):
    B, T_src = src_batch.shape
    u_e = jnp.zeros((B, d_e))
    s_e_prev = jnp.zeros((B, K_len, d_e))
    u_c = jnp.zeros((B, d_c))
    s_c_prev = jnp.zeros((B, K_len, d_c))

    def encode_scan(carry, x_t):
        u_e, s_e_prev, u_c, s_c_prev = carry
        s_x = jax.nn.one_hot(x_t, params['enc_w_e'].shape[1])
        s_e_new, s_c_new = [], []
        for tau in range(K_len):
            u_e, s_e_tau = encoder_step(u_e, s_e_prev[:, tau], s_x, params['enc_w_e'], params['enc_w_r'], beta_seq)
            u_c, s_c_tau = stcm_encoder_step(u_c, s_c_prev[:, tau], s_e_tau, params['stcm_w_ce'], params['stcm_w_cc'], beta_seq)
            s_e_new.append(s_e_tau)
            s_c_new.append(s_c_tau)
        return (u_e, jnp.stack(s_e_new, axis=1), u_c, jnp.stack(s_c_new, axis=1)), None

    src_batch_T = jnp.swapaxes(src_batch, 0, 1)
    (u_e, s_e_prev, u_c_ctx, s_c_ctx), _ = jax.lax.scan(encode_scan, (u_e, s_e_prev, u_c, s_c_prev), src_batch_T)
    return u_c_ctx, s_c_ctx

@functools.partial(jax.jit, static_argnums=(6,))
def decode_step_jax(params, current_tokens, u_c_ctx, s_c_ctx, u_d, s_d_prev, K_len, m_v_all):
    vocab_tgt = params['dec_w_y'].shape[1]
    s_y = jax.nn.one_hot(current_tokens, vocab_tgt)
    s_c_new, s_d_new = [], []
    for tau in range(K_len):
        u_c_ctx, s_c_tau = stcm_decoder_step(u_c_ctx, s_c_ctx[:, tau], s_d_prev[:, tau], params['stcm_w_ctx'], params['stcm_w_self'], beta_seq)
        u_d, s_d_tau = decoder_step(u_d, s_d_prev[:, tau], s_y, s_c_tau, params['dec_w_y'], params['dec_w_c'], params['dec_w_r'], beta_seq)
        s_c_new.append(s_c_tau)
        s_d_new.append(s_d_tau)

    s_c_ctx = jnp.stack(s_c_new, axis=1)
    s_d_prev = jnp.stack(s_d_new, axis=1)

    s_d_sum = jnp.sum(s_d_prev, axis=1)
    def calc_scores(batch_s_d_sum):
        return jnp.sum(batch_s_d_sum[m_v_all], axis=1)
    scores = jax.vmap(calc_scores)(s_d_sum)
    return u_c_ctx, s_c_ctx, u_d, s_d_prev, scores

def infer(params, src_batch, max_len, m_v_all, ngram_mem=None):
    B = src_batch.shape[0]
    vocab_tgt = params['dec_w_y'].shape[1]
    
    # 1. JIT Compiled Encoder Phase
    u_c_ctx, s_c_ctx = encode_batch(params, src_batch, K)
    
    # 2. Autoregressive Decoder Phase
    u_d = jnp.zeros((B, d_d))
    s_d_prev = jnp.zeros((B, K, d_d))
    current_tokens = jnp.full((B,), 2, dtype=jnp.int32)
    results = []
    
    for t in range(max_len):
        # Heavy computation on GPU via JIT
        u_c_ctx, s_c_ctx, u_d, s_d_prev, scores = decode_step_jax(
            params, current_tokens, u_c_ctx, s_c_ctx, u_d, s_d_prev, K, m_v_all
        )
        
        # Bring scores to CPU only for N-gram lookup
        scores = np.array(scores)
        for b in range(B):
            if ngram_mem:
                cands = ngram_mem.get_candidates(int(current_tokens[b]))
                for (tok, prob) in cands:
                    if tok < vocab_tgt:
                        scores[b, tok] += prob * 2.0
            scores[b, 0] = -np.inf
            scores[b, 1] = -np.inf
            
        next_tokens = jnp.argmax(scores, axis=1)
        results.append(next_tokens)
        current_tokens = next_tokens
        
    return jnp.stack(results, axis=1)

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
