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
def encoder_step(u_e, s_e_prev, s_x, w_e, w_r, beta_seq, threshold=1.0):
    u = u_e * beta_seq + jnp.dot(w_e, s_x) + jnp.dot(w_r, s_e_prev)
    s = spike_fn(u, threshold)
    u = u - s * threshold # Soft reset (differentiable)
    return u, s

# STCM Step
def stcm_encoder_step(u_c, s_c_prev, s_e, w_ce, w_cc, beta_seq, threshold=1.0):
    u = u_c * beta_seq + jnp.dot(w_ce, s_e) + jnp.dot(w_cc, s_c_prev)
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s

def stcm_decoder_step(u_c, s_c_prev, s_d, w_ctx, w_self, beta_seq, threshold=1.0):
    u = u_c * beta_seq + jnp.dot(w_ctx, s_d) + jnp.dot(w_self, s_c_prev)
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s

# Decoder Step
def decoder_step(u_d, s_d_prev, s_y, s_ctx, w_y, w_c, w_r, beta_seq, threshold=1.0):
    u = u_d * beta_seq + jnp.dot(w_y, s_y) + jnp.dot(w_c, s_ctx) + jnp.dot(w_r, s_d_prev)
    s = spike_fn(u, threshold)
    u = u - s * threshold
    return u, s
