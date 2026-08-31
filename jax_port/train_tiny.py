import jax
import jax.numpy as jnp
import numpy as np
from dataset import StreamingCorpus
from snn_core import precompute_all_sdr
from train import init_params, forward_pass, d_d, num_active_sdr
from infer import infer
import functools
import time

epochs = 10
max_samples = 100
batch_size = 10
learning_rate = 0.005

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
        g = jnp.clip(g, -1.0, 1.0)
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
    
    corpus = StreamingCorpus()
    vocab_src = corpus.vocab_size()
    vocab_tgt = corpus.vocab_size()
    
    params = init_params(vocab_src, vocab_tgt)
    # Initialize Adam state
    m = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    v = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
    
    params = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), params)
    m = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), m)
    v = jax.tree_util.tree_map(lambda x: jnp.stack([x] * num_devices), v)
    
    # Calculate Total vs Active Parameters
    total_params = sum(x[0].size for x in jax.tree_util.tree_leaves(params))
    # Active Params: Total params minus (E-1)*Expert Params
    from train import num_experts
    expert_params_per_expert = (params['exp_w1'][0].size + params['exp_w2'][0].size + 
                                params['exp1_g'][0].size + params['exp1_b'][0].size +
                                params['exp2_g'][0].size + params['exp2_b'][0].size) // num_experts
    active_params = total_params - (num_experts - 1) * expert_params_per_expert
    
    print(f"\n[Spiking-MoE Parameters]")
    print(f"Total Parameters : {total_params:,}")
    print(f"Active Parameters: {active_params:,} ({(active_params/total_params)*100:.1f}%)")
    print(f"Number of Experts: {num_experts}")
    print("-" * 30 + "\n")
    
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    global_step = 0
    for epoch in range(1, epochs + 1):
        batch_iter = corpus.stream_batches(batch_size, max_samples, max_seq_len=20)
        epoch_loss = 0.0
        steps = 0
        
        for src_batch_full, tgt_batch_full in batch_iter:
            src_batch = src_batch_full.reshape(num_devices, batch_size // num_devices, -1)
            tgt_batch = tgt_batch_full.reshape(num_devices, batch_size // num_devices, -1)
            
            global_step += 1
            steps += 1
            t = jnp.array([global_step] * num_devices, dtype=jnp.float32)
            params, loss, m, v = update_adam(params, src_batch, tgt_batch, m_v_all, m, v, t)
            epoch_loss += float(jnp.mean(loss))
            
        print(f"Epoch {epoch:3d}/{epochs} | Avg Loss = {epoch_loss / steps:.4f}")
            
    print("\nTraining selesai. Melakukan inferensi pada 5 data latih pertama (Overfit Test)...")
    
    # Ambil ulang 1 batch pertama untuk inferensi
    batch_iter = corpus.stream_batches(batch_size, max_samples, max_seq_len=20)
    src_batch_full, tgt_batch_full = next(batch_iter)
    
    saved_params = jax.tree_util.tree_map(lambda x: x[0], params)
    
    gen_len = tgt_batch_full.shape[1]
    
    start_time = time.time()
    out_ids, expert_ids = infer(saved_params, jnp.array(src_batch_full), max_len=gen_len, m_v_all=m_v_all)
    end_time = time.time()
    
    out_ids = np.array(out_ids)
    expert_ids = np.array(expert_ids)
    
    latency = end_time - start_time
    total_tokens_generated = batch_size * gen_len
    ms_per_token = (latency / total_tokens_generated) * 1000
    
    print(f"[Latency] {ms_per_token:.2f} ms/token (Total time: {latency:.2f}s for {total_tokens_generated} tokens)")
    
    # Histogram utilization
    valid_expert_mask = (src_batch_full != 0)
    valid_expert_ids = expert_ids[valid_expert_mask]
    hist = np.bincount(valid_expert_ids.flatten(), minlength=num_experts)
    print(f"[Router Histogram] Expert Load Distribution: {hist}")
    
    eos_id = corpus.get_eos_id()
    pad_id = corpus.get_pad_id()
    
    exact_matches = 0
    total_samples = batch_size
    
    for b in range(min(5, batch_size)):
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
        
        if tgt_text.strip() == pred_text.strip():
            exact_matches += 1
            
    print(f"\n[Accuracy] Exact Sequence Match: {exact_matches}/{total_samples} ({(exact_matches/total_samples)*100:.1f}%)")

if __name__ == "__main__":
    main()
