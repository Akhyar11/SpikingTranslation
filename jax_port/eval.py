import jax
import jax.numpy as jnp
import numpy as np
import time
from tqdm import tqdm
from dataset import StreamingCorpus
from snn_core import precompute_all_sdr
from train import init_params, d_d, num_active_sdr, num_experts
from infer import infer
import sacrebleu

def main():
    print("Memuat Vocabulary...")
    corpus = StreamingCorpus()
    vocab_src = corpus.vocab_size()
    vocab_tgt = corpus.vocab_size()
    
    print("Memuat Checkpoint Model...")
    try:
        ckpt = np.load("best_model_jax.npz")
        params = {k: jnp.array(v) for k, v in ckpt.items()}
    except:
        print("Checkpoint best_model_jax.npz tidak ditemukan. Anda harus training dulu.")
        return
        
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    # Hitung Parameters (Menjawab RQ3)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    expert_params_per_expert = (params['exp_w1'].size + params['exp_w2'].size + 
                                params['exp1_g'].size + params['exp1_b'].size +
                                params['exp2_g'].size + params['exp2_b'].size) // num_experts
    active_params = total_params - (num_experts - 1) * expert_params_per_expert
    
    print(f"\n======================================")
    print(f"[RQ3] Spiking-MoE Parameters")
    print(f"Total Parameters : {total_params:,}")
    print(f"Active Parameters: {active_params:,} ({(active_params/total_params)*100:.1f}%)")
    print(f"Number of Experts: {num_experts}")
    print(f"======================================\n")
    
    # Ambil 1000 data terakhir sebagai Test Set
    print("Menyiapkan 1000 data Test Set (OpenSubtitles 99.000 - 100.000)...")
    src_lines = []
    tgt_lines = []
    with open(corpus.src_path, 'r', encoding='utf-8') as f_src, open(corpus.tgt_path, 'r', encoding='utf-8') as f_tgt:
        for i, (s, t) in enumerate(zip(f_src, f_tgt)):
            if i >= 99000 and i < 100000:
                src_lines.append(s.strip())
                tgt_lines.append(t.strip())
            elif i >= 100000:
                break
                
    preds = []
    refs = [tgt_lines]
    
    batch_size = 32
    all_expert_ids = []
    total_latency = 0
    total_tokens = 0
    
    print("Memulai Inferensi dan Evaluasi...")
    for i in tqdm(range(0, len(src_lines), batch_size)):
        batch_src = src_lines[i:i+batch_size]
        
        # Padding & Encode
        pad_id = corpus.get_pad_id()
        eos_id = corpus.get_eos_id()
        
        encoded_src = [corpus.encode(s) + [eos_id] for s in batch_src]
        max_len = max(len(s) for s in encoded_src)
        src_padded = np.array([s + [pad_id]*(max_len - len(s)) for s in encoded_src], dtype=np.int32)
        
        start_time = time.time()
        out_ids, expert_ids = infer(params, jnp.array(src_padded), max_len=30, m_v_all=m_v_all)
        end_time = time.time()
        
        total_latency += (end_time - start_time)
        out_ids = np.array(out_ids)
        expert_ids = np.array(expert_ids)
        
        # Kumpulkan expert IDs untuk mask yang valid (bukan padding)
        valid_mask = (src_padded != pad_id)
        all_expert_ids.extend(expert_ids[valid_mask].flatten().tolist())
        
        for b in range(len(batch_src)):
            pred_seq = out_ids[b].tolist()
            if eos_id in pred_seq:
                pred_seq = pred_seq[:pred_seq.index(eos_id)]
            pred_text = corpus.decode(pred_seq).replace("<EOS>", "").strip()
            preds.append(pred_text)
            total_tokens += len(pred_seq)
            
    # Histogram (Menjawab RQ2)
    hist = np.bincount(all_expert_ids, minlength=num_experts)
    print(f"\n======================================")
    print(f"[RQ2] Router Load Balancing")
    print(f"Expert Load Distribution (Histogram):")
    print(hist)
    print(f"======================================\n")
    
    # BLEU Score (Menjawab RQ1)
    bleu = sacrebleu.corpus_bleu(preds, refs)
    chrf = sacrebleu.corpus_chrf(preds, refs)
    ms_per_token = (total_latency / total_tokens) * 1000 if total_tokens > 0 else 0
    
    print(f"\n======================================")
    print(f"[RQ1 & RQ3] Translation Quality & Latency")
    print(f"BLEU Score : {bleu.score:.2f}")
    print(f"chrF Score : {chrf.score:.2f}")
    print(f"Latency    : {ms_per_token:.2f} ms/token")
    print(f"======================================\n")

if __name__ == "__main__":
    main()
