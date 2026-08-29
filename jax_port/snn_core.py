import jax
import jax.numpy as jnp

# Fast Sigmoid Surrogate Derivative
@jax.custom_vjp
def spike_fn(u, threshold=1.0):
    return jnp.where(u >= threshold, 1.0, 0.0)

def spike_fn_fwd(u, threshold):
    return spike_fn(u, threshold), (u, threshold)

def spike_fn_bwd(res, g):
    u, threshold = res
    gamma = 1.0
    diff = u - threshold
    grad_u = gamma / (1.0 + 5.0 * jnp.abs(diff))**2
    return (g * grad_u, None)

spike_fn.defvjp(spike_fn_fwd, spike_fn_bwd)

# SDR Token Map
def sdr_token_map(v, d_d, num_active=3):
    """
    Deterministic Hash Mapping for tokens.
    Returns an array of indices [num_active] representing the token.
    """
    import numpy as np
    # Kita menggunakan numpy untuk precomputation map ini karena tidak perlu di-diferensiasi.
    seed = (v * 1234567891 + 987654321) % (2**64)
    indices = set()
    while len(indices) < num_active:
        seed = seed ^ ((seed << 13) % (2**64))
        seed = seed ^ (seed >> 17)
        seed = seed ^ ((seed << 5) % (2**64))
        indices.add(int(seed % d_d))
    return np.array(list(indices), dtype=np.int32)

def precompute_all_sdr(vocab_size, d_d, num_active=3):
    import numpy as np
    m_v_all = np.zeros((vocab_size, num_active), dtype=np.int32)
    for v in range(vocab_size):
        m_v_all[v] = sdr_token_map(v, d_d, num_active)
    return jnp.array(m_v_all)

# Encoder Step
def encoder_step(u_e, s_e_prev, s_x, w_e, w_r, gamma, beta, beta_seq, threshold=0.1):
    syn_input = jnp.dot(s_x, w_e.T) + jnp.dot(s_e_prev, w_r.T)
    mean = jnp.mean(syn_input, axis=-1, keepdims=True)
    var = jnp.var(syn_input, axis=-1, keepdims=True)
    syn_norm = gamma * (syn_input - mean) / jnp.sqrt(var + 1e-5) + beta
    
    u = u_e * beta_seq + syn_norm
    s = spike_fn(u, threshold)
    u = u - s * threshold # Soft reset (differentiable)
    return u, s

# Spiking Router Step (Phase 3)
def router_step(u_router, s_e_tau, w_router, gamma, beta_norm, beta_seq, threshold=0.1):
    syn_input = jnp.dot(s_e_tau, w_router.T)
    mean = jnp.mean(syn_input, axis=-1, keepdims=True)
    var = jnp.var(syn_input, axis=-1, keepdims=True)
    syn_norm = gamma * (syn_input - mean) / jnp.sqrt(var + 1e-5) + beta_norm
    
    u = u_router * beta_seq + syn_norm
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s

# Dense Expert Step (Phase 1 & 3)
def dense_expert_step(u_h, u_o, s_in, w1, w2, gamma1, beta1, gamma2, beta2, beta_seq, threshold=0.1):
    # Layer 1
    if w1.ndim == 3:
        proj_1 = jnp.einsum('be, bhe -> bh', s_in, w1)
    else:
        proj_1 = jnp.dot(s_in, w1.T)
        
    mean1 = jnp.mean(proj_1, axis=-1, keepdims=True)
    var1 = jnp.var(proj_1, axis=-1, keepdims=True)
    proj_1_norm = gamma1 * (proj_1 - mean1) / jnp.sqrt(var1 + 1e-5) + beta1
        
    u_h = u_h * beta_seq + proj_1_norm
    s_h = spike_fn(u_h, threshold)
    u_h = u_h - s_h * threshold
    
    # Layer 2
    if w2.ndim == 3:
        proj_2 = jnp.einsum('bh, beh -> be', s_h, w2)
    else:
        proj_2 = jnp.dot(s_h, w2.T)
        
    mean2 = jnp.mean(proj_2, axis=-1, keepdims=True)
    var2 = jnp.var(proj_2, axis=-1, keepdims=True)
    proj_2_norm = gamma2 * (proj_2 - mean2) / jnp.sqrt(var2 + 1e-5) + beta2
        
    u_o = u_o * beta_seq + proj_2_norm
    s_o = spike_fn(u_o, threshold)
    u_o = u_o - s_o * threshold
    
    return u_h, u_o, s_o

# STCM Step
def stcm_encoder_step(u_c, s_c_prev, s_e, w_ce, w_cc, gamma, beta, beta_seq, threshold=0.1):
    syn_input = jnp.dot(s_e, w_ce.T) + jnp.dot(s_c_prev, w_cc.T)
    mean = jnp.mean(syn_input, axis=-1, keepdims=True)
    var = jnp.var(syn_input, axis=-1, keepdims=True)
    syn_norm = gamma * (syn_input - mean) / jnp.sqrt(var + 1e-5) + beta
    
    u = u_c * beta_seq + syn_norm
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s

def stcm_decoder_step(u_c, s_c_prev, s_d, w_ctx, w_self, gamma, beta, beta_seq, threshold=0.1):
    syn_input = jnp.dot(s_d, w_ctx.T) + jnp.dot(s_c_prev, w_self.T)
    mean = jnp.mean(syn_input, axis=-1, keepdims=True)
    var = jnp.var(syn_input, axis=-1, keepdims=True)
    syn_norm = gamma * (syn_input - mean) / jnp.sqrt(var + 1e-5) + beta
    
    u = u_c * beta_seq + syn_norm
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s

# Decoder Step
def decoder_step(u_d, s_d_prev, s_y, s_ctx, w_y, w_c, w_r, gamma, beta, beta_seq, threshold=0.1):
    syn_input = jnp.dot(s_y, w_y.T) + jnp.dot(s_ctx, w_c.T) + jnp.dot(s_d_prev, w_r.T)
    mean = jnp.mean(syn_input, axis=-1, keepdims=True)
    var = jnp.var(syn_input, axis=-1, keepdims=True)
    syn_norm = gamma * (syn_input - mean) / jnp.sqrt(var + 1e-5) + beta
    
    u = u_d * beta_seq + syn_norm
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s
