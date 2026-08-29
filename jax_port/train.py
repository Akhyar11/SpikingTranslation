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
K = 20 # Neural timesteps
beta_seq = 0.1
learning_rate = 0.01
batch_size = 32
max_samples = 99000 # Sisakan 1000 terakhir (99k-100k) untuk test set
epochs = 10
d_e = 64
d_c = 128
d_d = 128
num_active_sdr = 3
margin = 1.0

def init_params(d_in_src, d_in_tgt):
    key = jax.random.PRNGKey(42)
    def uniform(k, shape):
        return jax.random.uniform(k, shape, minval=-0.5, maxval=0.5) * 0.1
        
    keys = jax.random.split(key, 10)
    params = {
        'enc_w_e': uniform(keys[0], (d_e, d_in_src)),
        'enc_w_r': uniform(keys[1], (d_e, d_e)) + jnp.eye(d_e) * 0.9,
        'stcm_w_ce': uniform(keys[2], (d_c, d_e)),
        'stcm_w_cc': uniform(keys[3], (d_c, d_c)) + jnp.eye(d_c) * 0.9,
        'stcm_w_ctx': uniform(keys[4], (d_c, d_d)),
        'stcm_w_self': uniform(keys[5], (d_c, d_c)) + jnp.eye(d_c) * 0.9,
        'dec_w_y': uniform(keys[6], (d_d, d_in_tgt)),
        'dec_w_c': uniform(keys[7], (d_d, d_c)),
        'dec_w_r': uniform(keys[8], (d_d, d_d)) + jnp.eye(d_d) * 0.9
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
    u_c_enc = jnp.zeros((B, d_c))
    s_c_enc_prev = jnp.zeros((B, K, d_c))
    
    # --- ENCODER PHASE ---
    def encode_scan(carry, x_t):
        u_e, s_e_prev, u_c, s_c_prev = carry
        # x_t: (B,)
        # One-hot representation inside the scan
        s_x = jax.nn.one_hot(x_t, params['enc_w_e'].shape[1]) # (B, vocab_src)
        
        s_e_new = []
        s_c_new = []
        
        for tau in range(K):
            u_e, s_e_tau = encoder_step(u_e, s_e_prev[:, tau], s_x, params['enc_w_e'], params['enc_w_r'], beta_seq)
            u_c, s_c_tau = stcm_encoder_step(u_c, s_c_prev[:, tau], s_e_tau, params['stcm_w_ce'], params['stcm_w_cc'], beta_seq)
            s_e_new.append(s_e_tau)
            s_c_new.append(s_c_tau)
            
        s_e_prev = jnp.stack(s_e_new, axis=1) # (B, K, d_e)
        s_c_prev = jnp.stack(s_c_new, axis=1) # (B, K, d_c)
        return (u_e, s_e_prev, u_c, s_c_prev), None

    # Scan over source tokens
    src_batch_T = jnp.swapaxes(src_batch, 0, 1) # (T_src, B)
    (u_e, s_e_prev, u_c_ctx, s_c_ctx), _ = jax.lax.scan(encode_scan, (u_e, s_e_prev, u_c_enc, s_c_enc_prev), src_batch_T)
    
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
            u_c_ctx, s_c_tau = stcm_decoder_step(u_c_ctx, s_c_ctx[:, tau], s_d_prev[:, tau], params['stcm_w_ctx'], params['stcm_w_self'], beta_seq)
            u_d, s_d_tau = decoder_step(u_d, s_d_prev[:, tau], s_y, s_c_tau, params['dec_w_y'], params['dec_w_c'], params['dec_w_r'], beta_seq)
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
        
        # score for token v: sum(s_d_sum[m_v_all[v]])
        # We can vmap this or use advanced indexing
        def calc_scores(batch_s_d_sum):
            # batch_s_d_sum: (d_d,)
            return jnp.sum(batch_s_d_sum[m_v_all], axis=1) # (vocab_tgt,)
            
        scores = jax.vmap(calc_scores)(s_d_sum) # (B, vocab_tgt)
        
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
    return jnp.sum(losses) / B

@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None))
def update(params, src_batch, tgt_batch, m_v_all):
    loss, grads = jax.value_and_grad(forward_pass)(params, src_batch, tgt_batch, m_v_all)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return new_params, loss

def main():
    print("Mulai Pelatihan SNN (JAX Port)...")
    num_devices = jax.local_device_count()
    print(f"Ditemukan {num_devices} device(s) JAX. Menggunakan Data Parallelism (@jax.pmap).")
    
    corpus = StreamingCorpus("../dataset/OpenSubtitles.en-id.en", "../dataset/OpenSubtitles.en-id.id")
    vocab_src = corpus.vocab_size()
    vocab_tgt = corpus.vocab_size()
    
    params = init_params(vocab_src, vocab_tgt)
    # Duplikasi parameter ke memori setiap GPU
    params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
    
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
            
            params, loss = update(params, src_batch, tgt_batch, m_v_all)
            total_loss += float(jnp.mean(loss))
            steps += 1
            pbar.set_postfix({"Loss": f"{total_loss/steps:.4f}"})

        # Save checkpoint (hanya simpan versi GPU 0 agar tidak redundant)
        saved_params = jax.tree_util.tree_map(lambda x: x[0], params)
        np.savez("best_model_jax.npz", **saved_params)
        print(f"Epoch {epoch} selesai. Checkpoint disimpan ke best_model_jax.npz")

if __name__ == "__main__":
    main()
