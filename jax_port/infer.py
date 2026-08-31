import jax
import jax.numpy as jnp
import numpy as np
import functools
from dataset import StreamingCorpus, SparseNGramMemory
from snn_core import (
    encoder_step, stcm_encoder_step, stcm_decoder_step, decoder_step,
    precompute_all_sdr
)

from train import K, beta_seq, d_d, d_e, d_hidden, d_c
num_active_sdr = 3

@functools.partial(jax.jit, static_argnums=(2,))
def encode_batch(params, src_batch, K_len):
    B, T_src = src_batch.shape
    u_e = jnp.zeros((B, d_e))
    s_e_prev = jnp.zeros((B, K_len, d_e))
    u_router = jnp.zeros((B, params['exp_w_router'].shape[0]))
    u_exp_h = jnp.zeros((B, d_hidden))
    u_exp_o = jnp.zeros((B, d_e))
    u_c = jnp.zeros((B, d_c))
    s_c_prev = jnp.zeros((B, K_len, d_c))

    def encode_scan(carry, x_t):
        u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c, s_c_prev = carry
        s_x = jax.nn.one_hot(x_t, params['enc_w_e'].shape[1])
        s_e_new, s_c_new, s_router_new = [], [], []
        u_e_next, u_router_next = u_e, u_router
        u_exp_h_next, u_exp_o_next, u_c_next = u_exp_h, u_exp_o, u_c
        s_e_prev_next, s_c_prev_next = s_e_prev, s_c_prev
        
        from snn_core import dense_expert_step, router_step
        
        # Loop 1: Encoder & Router
        for tau in range(K_len):
            u_e_next, s_e_tau = encoder_step(u_e_next, s_e_prev_next[:, tau], s_x, params['enc_w_e'], params['enc_w_r'], params['enc_g'], params['enc_b'], beta_seq)
            u_router_next, s_router_tau = router_step(u_router_next, s_e_tau, params['exp_w_router'], params['router_g'], params['router_b'], beta_seq)
            s_e_new.append(s_e_tau)
            s_router_new.append(s_router_tau)
            
        s_router_stack = jnp.stack(s_router_new, axis=1)
        R_e = jnp.sum(s_router_stack, axis=1)
        
        # Tie-breaker (voltage sub-threshold) to prevent integer argmax index 0 collapse
        router_score = R_e + u_router_next * 0.01
        expert_id = jnp.argmax(router_score, axis=1)
        
        w1_expert = params['exp_w1'][expert_id]
        w2_expert = params['exp_w2'][expert_id]
        g1_expert = params['exp1_g'][expert_id]
        b1_expert = params['exp1_b'][expert_id]
        g2_expert = params['exp2_g'][expert_id]
        b2_expert = params['exp2_b'][expert_id]
        
        # Loop 2: Expert & STCM
        for tau in range(K_len):
            u_exp_h_next, u_exp_o_next, s_exp_tau = dense_expert_step(u_exp_h_next, u_exp_o_next, s_e_new[tau], w1_expert, w2_expert, g1_expert, b1_expert, g2_expert, b2_expert, beta_seq)
            u_c_next, s_c_tau = stcm_encoder_step(u_c_next, s_c_prev_next[:, tau], s_exp_tau, params['stcm_w_ce'], params['stcm_w_cc'], params['stcm_enc_g'], params['stcm_enc_b'], beta_seq)
            s_c_new.append(s_c_tau)
            
        s_e_new_stack = jnp.stack(s_e_new, axis=1)
        s_c_new_stack = jnp.stack(s_c_new, axis=1)
        
        is_valid = (x_t != 0).reshape(-1, 1)
        is_valid_s = is_valid.reshape(-1, 1, 1)
        
        u_e_final = jnp.where(is_valid, u_e_next, u_e)
        u_router_final = jnp.where(is_valid, u_router_next, u_router)
        u_exp_h_final = jnp.where(is_valid, u_exp_h_next, u_exp_h)
        u_exp_o_final = jnp.where(is_valid, u_exp_o_next, u_exp_o)
        u_c_final = jnp.where(is_valid, u_c_next, u_c)
        s_e_final = jnp.where(is_valid_s, s_e_new_stack, s_e_prev)
        s_c_final = jnp.where(is_valid_s, s_c_new_stack, s_c_prev)
        
        return (u_e_final, s_e_final, u_router_final, u_exp_h_final, u_exp_o_final, u_c_final, s_c_final), expert_id

    src_batch_T = jnp.swapaxes(src_batch, 0, 1)
    (u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c_ctx, s_c_ctx), expert_ids_T = jax.lax.scan(encode_scan, (u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c, s_c_prev), src_batch_T)
    return u_c_ctx, s_c_ctx, jnp.swapaxes(expert_ids_T, 0, 1)

@functools.partial(jax.jit, static_argnums=(6,))
def decode_step_jax(params, current_tokens, u_c_ctx, s_c_ctx, u_d, s_d_prev, K_len, m_v_all):
    vocab_tgt = params['dec_w_y'].shape[1]
    s_y = jax.nn.one_hot(current_tokens, vocab_tgt)
    s_c_new, s_d_new = [], []
    for tau in range(K_len):
        u_c_ctx, s_c_tau = stcm_decoder_step(u_c_ctx, s_c_ctx[:, tau], s_d_prev[:, tau], params['stcm_w_ctx'], params['stcm_w_self'], params['stcm_dec_g'], params['stcm_dec_b'], beta_seq)
        u_d, s_d_tau = decoder_step(u_d, s_d_prev[:, tau], s_y, s_c_tau, params['dec_w_y'], params['dec_w_c'], params['dec_w_r'], params['dec_g'], params['dec_b'], beta_seq)
        s_c_new.append(s_c_tau)
        s_d_new.append(s_d_tau)

    s_c_ctx = jnp.stack(s_c_new, axis=1)
    s_d_prev = jnp.stack(s_d_new, axis=1)

    s_d_sum = jnp.sum(s_d_prev, axis=1)
    
    def calc_scores(batch_s, batch_u):
        spike_score = jnp.sum(batch_s[m_v_all], axis=1)
        volt_score = jnp.sum(batch_u[m_v_all], axis=1) * 0.05
        return spike_score + volt_score
        
    scores = jax.vmap(calc_scores)(s_d_sum, u_d)
    return u_c_ctx, s_c_ctx, u_d, s_d_prev, scores

def infer(params, src_batch, max_len=64, m_v_all=None):
    B = src_batch.shape[0]
    vocab_tgt = params['dec_w_y'].shape[1]
    
    # 1. JIT Compiled Encoder Phase
    u_c_ctx, s_c_ctx, expert_ids = encode_batch(params, src_batch, K)
    
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
        
        # JAX array to masked JAX array (avoiding numpy conversion)
        scores = scores.at[:, :2].set(-jnp.inf)
            
        next_tokens = jnp.argmax(scores, axis=1)
        results.append(next_tokens)
        current_tokens = next_tokens
        
    return jnp.stack(results, axis=1), expert_ids

def main():
    print("Memuat Vocabulary...")
    corpus = StreamingCorpus()
    vocab_tgt = corpus.vocab_size()
    
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
        out_ids = infer(params, src_batch, max_len=20, m_v_all=m_v_all)
        out_text = corpus.decode(out_ids[0].tolist())
        print("En:", text)
        print("Id:", out_text.replace("<EOS>", "").strip())
        print("------------------")

if __name__ == "__main__":
    main()
