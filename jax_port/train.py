import jax
import jax.numpy as jnp
import numpy as np
import functools
from tqdm import tqdm
from dataset import StreamingCorpus
from snn_core import (
    encoder_step, stcm_encoder_step, stcm_decoder_step, decoder_step,
    precompute_all_sdr
)

# Hyperparameters
K = 5 # Neural timesteps
beta_seq = 0.9
learning_rate = 0.01
batch_size = 32
max_samples = 99000 # Sisakan 1000 terakhir (99k-100k) untuk test set
epochs = 10
d_e = 64
d_hidden = 1024
d_c = 128
d_d = 128
num_active_sdr = 3
margin = 1.0
num_experts = 128

def init_params(d_in_src, d_in_tgt):
    key = jax.random.PRNGKey(42)
    def uniform(k, shape):
        scale = 1.0 / jnp.sqrt(shape[-1])
        return jax.random.uniform(k, shape, minval=-scale, maxval=scale)
        
    keys = jax.random.split(key, 12)
    params = {
        'enc_w_e': uniform(keys[0], (d_e, d_in_src)),
        'enc_w_r': uniform(keys[1], (d_e, d_e)),
        'exp_w1': uniform(keys[2], (num_experts, d_hidden, d_e)),
        'exp_w2': uniform(keys[3], (num_experts, d_e, d_hidden)),
        'stcm_w_ce': uniform(keys[4], (d_c, d_e)),
        'stcm_w_cc': uniform(keys[5], (d_c, d_c)),
        'stcm_w_ctx': uniform(keys[6], (d_c, d_d)),
        'stcm_w_self': uniform(keys[7], (d_c, d_c)),
        'dec_w_y': uniform(keys[8], (d_d, d_in_tgt)),
        'dec_w_c': uniform(keys[9], (d_d, d_c)),
        'dec_w_r': uniform(keys[10], (d_d, d_d)),
        'exp_w_router': uniform(keys[11], (num_experts, d_e)),
        
        # LayerNorm parameters
        'enc_g': jnp.ones((d_e,)), 'enc_b': jnp.zeros((d_e,)),
        'router_g': jnp.ones((num_experts,)), 'router_b': jnp.zeros((num_experts,)),
        'exp1_g': jnp.ones((num_experts, d_hidden)), 'exp1_b': jnp.zeros((num_experts, d_hidden)),
        'exp2_g': jnp.ones((num_experts, d_e)), 'exp2_b': jnp.zeros((num_experts, d_e)),
        'stcm_enc_g': jnp.ones((d_c,)), 'stcm_enc_b': jnp.zeros((d_c,)),
        'stcm_dec_g': jnp.ones((d_c,)), 'stcm_dec_b': jnp.zeros((d_c,)),
        'dec_g': jnp.ones((d_d,)), 'dec_b': jnp.zeros((d_d,))
    }
    return params

@jax.jit
def forward_pass(params, src_batch, tgt_batch, m_v_all):
    # src_batch: (B, T_src)
    # tgt_batch: (B, T_tgt)
    B, T_src = src_batch.shape
    _, T_tgt = tgt_batch.shape
    vocab_tgt = params['dec_w_y'].shape[1]
    
    # Init states
    u_e = jnp.zeros((B, d_e))
    s_e_prev = jnp.zeros((B, K, d_e))
    u_router = jnp.zeros((B, num_experts))
    u_exp_h = jnp.zeros((B, d_hidden))
    u_exp_o = jnp.zeros((B, d_e))
    u_c_enc = jnp.zeros((B, d_c))
    s_c_enc_prev = jnp.zeros((B, K, d_c))
    
    # --- ENCODER PHASE ---
    def encode_scan(carry, x_t):
        u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c, s_c_prev = carry
        s_x = jax.nn.one_hot(x_t, params['enc_w_e'].shape[1])
        
        s_e_new = []
        s_c_new = []
        s_router_new = []
        
        u_e_next, u_router_next = u_e, u_router
        u_exp_h_next, u_exp_o_next, u_c_next = u_exp_h, u_exp_o, u_c
        s_e_prev_next, s_c_prev_next = s_e_prev, s_c_prev
        
        # Loop 1: Encoder & Router
        from snn_core import dense_expert_step, router_step
        for tau in range(K):
            u_e_next, s_e_tau = encoder_step(u_e_next, s_e_prev_next[:, tau], s_x, params['enc_w_e'], params['enc_w_r'], params['enc_g'], params['enc_b'], beta_seq)
            u_router_next, s_router_tau = router_step(u_router_next, s_e_tau, params['exp_w_router'], params['router_g'], params['router_b'], beta_seq)
            s_e_new.append(s_e_tau)
            s_router_new.append(s_router_tau)
            
        # Accumulate spikes to select expert
        s_router_stack = jnp.stack(s_router_new, axis=1) # (B, K, E)
        R_e = jnp.sum(s_router_stack, axis=1) # (B, E)
        expert_id = jnp.argmax(R_e, axis=1) # (B,)
        
        w1_expert = params['exp_w1'][expert_id] # (B, d_hidden, d_e)
        w2_expert = params['exp_w2'][expert_id] # (B, d_e, d_hidden)
        g1_expert = params['exp1_g'][expert_id]
        b1_expert = params['exp1_b'][expert_id]
        g2_expert = params['exp2_g'][expert_id]
        b2_expert = params['exp2_b'][expert_id]
        
        # Loop 2: Selected Expert & STCM
        for tau in range(K):
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
        
        # Phase 4: CV-Squared Load Balancing Loss
        mean_router_spikes = jnp.mean(R_e, axis=0) # (E,)
        mean_of_means = jnp.mean(mean_router_spikes)
        var_of_means = jnp.var(mean_router_spikes)
        l_balance_t = var_of_means / (jnp.square(mean_of_means) + 1e-5)
        # Apply mask
        l_balance_t = jnp.where(x_t != 0, l_balance_t, 0.0)
        
        # Phase 5: Router Activity Loss (prevent zero-spike collapse)
        total_spikes = jnp.sum(R_e, axis=-1) # (B,)
        l_activity_t = jnp.mean(jnp.maximum(0.0, 1.0 - total_spikes))
        l_activity_t = jnp.where(x_t != 0, l_activity_t, 0.0)
        
        return (u_e_final, s_e_final, u_router_final, u_exp_h_final, u_exp_o_final, u_c_final, s_c_final), (l_balance_t, l_activity_t)

    # Scan over source tokens
    src_batch_T = jnp.swapaxes(src_batch, 0, 1) # (T_src, B)
    (u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c_ctx, s_c_ctx), (encoder_balances, encoder_activities) = jax.lax.scan(encode_scan, (u_e, s_e_prev, u_router, u_exp_h, u_exp_o, u_c_enc, s_c_enc_prev), src_batch_T)
    
    total_l_balance = jnp.sum(encoder_balances)
    total_l_activity = jnp.sum(encoder_activities)
    
    # --- DECODER PHASE ---
    # We use teacher forcing, so we feed tgt_batch[:, t-1] to predict tgt_batch[:, t]
    # For simplicity in this port, we feed <BOS> (id=2) followed by tgt_batch[:-1]
    bos_col = jnp.full((B, 1), 2, dtype=jnp.int32)
    tgt_inputs = jnp.concatenate([bos_col, tgt_batch[:, :-1]], axis=1)
    
    u_d = jnp.zeros((B, d_d))
    s_d_prev = jnp.zeros((B, K, d_d))
    
    def decode_scan(carry, inputs):
        u_c_ctx, s_c_ctx, u_d, s_d_prev = carry
        y_in, y_true = inputs # y_in: (B,), y_true: (B,)
        s_y = jax.nn.one_hot(y_in, vocab_tgt) # (B, vocab_tgt)
        
        s_c_new = []
        s_d_new = []
        u_d_history = []
        
        for tau in range(K):
            u_c_ctx, s_c_tau = stcm_decoder_step(u_c_ctx, s_c_ctx[:, tau], s_d_prev[:, tau], params['stcm_w_ctx'], params['stcm_w_self'], params['stcm_dec_g'], params['stcm_dec_b'], beta_seq)
            u_d, s_d_tau = decoder_step(u_d, s_d_prev[:, tau], s_y, s_c_tau, params['dec_w_y'], params['dec_w_c'], params['dec_w_r'], params['dec_g'], params['dec_b'], beta_seq)
            s_c_new.append(s_c_tau)
            s_d_new.append(s_d_tau)
            u_d_history.append(u_d)
            
        s_c_ctx = jnp.stack(s_c_new, axis=1)
        s_d_prev = jnp.stack(s_d_new, axis=1)
        s_d_t = s_d_prev # (B, K, d_d)
        
        # Calculate Margin Spike Loss
        # m_v_all is (vocab_tgt, 3) mapping tokens to 3 neurons
        # We need to compute score for all candidates: sum of spikes over K for the mapped neurons
        s_d_sum = jnp.sum(s_d_t, axis=1) # (B, d_d)
        
        # Add sub-threshold voltage (u_d) as a continuous tie-breaker and direct gradient path
        def calc_scores(batch_s, batch_u):
            spike_score = jnp.sum(batch_s[m_v_all], axis=1)
            volt_score = jnp.sum(batch_u[m_v_all], axis=1) * 0.05
            return spike_score + volt_score
            
        scores = jax.vmap(calc_scores)(s_d_sum, u_d) # (B, vocab_tgt)
        
        # a_plus (target score)
        a_plus = jnp.take_along_axis(scores, jnp.expand_dims(y_true, 1), axis=1).squeeze(1) # (B,)
        
        # a_minus (best negative score)
        # We mask out the true token
        scores_masked = jnp.where(jax.nn.one_hot(y_true, vocab_tgt) > 0, -jnp.inf, scores)
        a_minus = jnp.max(scores_masked, axis=1) # (B,)
        
        # Loss = max(0, margin - a_plus + a_minus)
        # Ignore loss if y_true is padding (assume pad_id=0)
        loss = jnp.where(y_true != 0, jnp.maximum(0.0, margin - a_plus + a_minus), 0.0)
        
        return (u_c_ctx, s_c_ctx, u_d, s_d_prev), jnp.sum(loss)

    tgt_inputs_T = jnp.swapaxes(tgt_inputs, 0, 1) # (T_tgt, B)
    tgt_true_T = jnp.swapaxes(tgt_batch, 0, 1)
    
    _, losses = jax.lax.scan(decode_scan, (u_c_ctx, s_c_ctx, u_d, s_d_prev), (tgt_inputs_T, tgt_true_T))
    
    task_loss = jnp.sum(losses) / B
    lambda_balance = 0.1
    lambda_activity = 0.5
    balance_loss = (total_l_balance / T_src) * lambda_balance
    activity_loss = (total_l_activity / T_src) * lambda_activity
    return task_loss + balance_loss + activity_loss

@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None, 0, 0, None))
def update(params, src_batch, tgt_batch, m_v_all, m, v, t):
    loss, grads = jax.value_and_grad(forward_pass)(params, src_batch, tgt_batch, m_v_all)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    def apply_adam(p, g, m_i, v_i):
        m_new = beta1 * m_i + (1 - beta1) * g
        v_new = beta2 * v_i + (1 - beta2) * jnp.square(g)
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        p_new = p - learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)
        return p_new, m_new, v_new

    new_params = {}
    new_m = {}
    new_v = {}
    
    for k in params.keys():
        p_new, m_new, v_new = apply_adam(params[k], grads[k], m[k], v[k])
        new_params[k] = p_new
        new_m[k] = m_new
        new_v[k] = v_new
        
    return new_params, loss, new_m, new_v

def main():
    print("Mulai Pelatihan SNN (JAX Port)...")
    num_devices = jax.local_device_count()
    print(f"Ditemukan {num_devices} device(s) JAX. Menggunakan Data Parallelism (@jax.pmap).")
    
    corpus = StreamingCorpus()
    vocab_src = corpus.vocab_size()
    vocab_tgt = corpus.vocab_size()
    
    params = init_params(vocab_src, vocab_tgt)
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    
    # Duplikasi parameter ke memori setiap GPU
    params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
    m = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), m)
    v = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), v)
    
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        batch_iter = corpus.stream_batches(batch_size, max_samples)
        total_loss = 0.0
        steps = 0
        
        pbar = tqdm(batch_iter, total=max_samples//batch_size)
        for src_batch, tgt_batch in pbar:
            # Pastikan bisa dibagi rata ke semua GPU
            if batch_size % num_devices != 0:
                raise ValueError("Batch size harus habis dibagi jumlah GPU!")
                
            src_batch = src_batch.reshape(num_devices, batch_size // num_devices, -1)
            tgt_batch = tgt_batch.reshape(num_devices, batch_size // num_devices, -1)
            
            steps += 1
            t = jnp.array([steps] * num_devices, dtype=jnp.float32)
            params, loss, m, v = update(params, src_batch, tgt_batch, m_v_all, m, v, t)
            
            total_loss += float(jnp.mean(loss))
            pbar.set_postfix({"Loss": f"{total_loss/steps:.4f}"})

        # Save checkpoint (hanya simpan versi GPU 0 agar tidak redundant)
        saved_params = jax.tree_util.tree_map(lambda x: x[0], params)
        np.savez("best_model_jax.npz", **saved_params)
        print(f"Epoch {epoch} selesai. Checkpoint disimpan ke best_model_jax.npz")

if __name__ == "__main__":
    main()
