import jax
import jax.numpy as jnp
import numpy as np
from dataset import StreamingCorpus
from snn_core import precompute_all_sdr
from train import init_params, forward_pass, d_d, num_active_sdr
from infer import infer
import functools

epochs = 300
max_samples = 20
batch_size = 20
learning_rate = 0.01

@functools.partial(jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None, 0, 0, None))
def update_adam(params, src_batch, tgt_batch, m_v_all, m, v, t):
    loss, grads = jax.value_and_grad(forward_pass)(params, src_batch, tgt_batch, m_v_all)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Adam Update
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
    print("Mulai Pelatihan SNN Tiny (Overfitting Test dengan Adam)...")
    num_devices = jax.local_device_count()
    
    corpus = StreamingCorpus("../dataset/OpenSubtitles.en-id.en", "../dataset/OpenSubtitles.en-id.id")
    vocab_src = corpus.vocab_size()
    vocab_tgt = corpus.vocab_size()
    
    params = init_params(vocab_src, vocab_tgt)
    # Initialize Adam state
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    
    params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
    m = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), m)
    v = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), v)
    
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    batch_iter = corpus.stream_batches(batch_size, max_samples, max_seq_len=20)
    src_batch_full, tgt_batch_full = next(batch_iter)
    
    for epoch in range(1, epochs + 1):
        src_batch = src_batch_full.reshape(num_devices, batch_size // num_devices, -1)
        tgt_batch = tgt_batch_full.reshape(num_devices, batch_size // num_devices, -1)
        
        t = jnp.array([epoch] * num_devices, dtype=jnp.float32)
        params, loss, m, v = update_adam(params, src_batch, tgt_batch, m_v_all, m, v, t)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss = {float(jnp.mean(loss)):.4f}")
            if float(jnp.mean(loss)) < 0.1:
                print("Loss sudah sangat kecil, early stopping!")
                break
            
    print("\nTraining selesai. Melakukan inferensi pada data latih (Overfit Test)...")
    
    saved_params = jax.tree_util.tree_map(lambda x: x[0], params)
    
    gen_len = tgt_batch_full.shape[1]
    out_ids = infer(saved_params, jnp.array(src_batch_full), max_len=gen_len, m_v_all=m_v_all, ngram_mem=None)
    out_ids = np.array(out_ids)
    
    eos_id = corpus.get_eos_id()
    pad_id = corpus.get_pad_id()
    
    for b in range(5):
        src_seq = src_batch_full[b].tolist()
        src_text = corpus.decode([x for x in src_seq if x not in (pad_id, eos_id)])
        
        tgt_seq = tgt_batch_full[b].tolist()
        tgt_text = corpus.decode([x for x in tgt_seq if x not in (pad_id, eos_id)])
        
        pred_seq = out_ids[b].tolist()
        if eos_id in pred_seq:
            pred_seq = pred_seq[:pred_seq.index(eos_id)]
        pred_text = corpus.decode(pred_seq)
        
        print(f"\n[{b+1}] Source : {src_text}")
        print(f"    Target : {tgt_text}")
        print(f"    Predik : {pred_text}")

if __name__ == "__main__":
    main()
