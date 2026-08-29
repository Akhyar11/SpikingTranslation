import jax
import jax.numpy as jnp
import numpy as np
import time
import sacrebleu
from tqdm import tqdm
from dataset import StreamingCorpus, SparseNGramMemory
from snn_core import precompute_all_sdr
from infer import infer # Reuse the autoregressive decoding from infer.py

d_d = 128
num_active_sdr = 3

def read_test_set(src_path, tgt_path, start_line=74000, end_line=75000):
    src_lines = []
    tgt_lines = []
    with open(src_path, 'r', encoding='utf-8') as f_src, open(tgt_path, 'r', encoding='utf-8') as f_tgt:
        for i, (s, t) in enumerate(zip(f_src, f_tgt)):
            if i >= start_line and i < end_line:
                src_lines.append(s.strip())
                tgt_lines.append(t.strip())
            if i >= end_line:
                break
    return src_lines, tgt_lines

def evaluate_model(model_name, params, corpus, src_texts, refs, m_v_all, ngram_mem, batch_size=32):
    print(f"\nMenjalankan Evaluasi: {model_name}")
    predictions = []
    
    start_time = time.time()
    total_tokens_generated = 0
    
    for i in tqdm(range(0, len(src_texts), batch_size)):
        batch_texts = src_texts[i:i+batch_size]
        
        # Encode
        batch_ids = []
        for txt in batch_texts:
            encoded = corpus.encode(txt)
            batch_ids.append(encoded)
            
        max_len_batch = max(len(x) for x in batch_ids)
        pad_id = corpus.get_pad_id()
        src_padded = np.array([x + [pad_id]*(max_len_batch - len(x)) for x in batch_ids], dtype=np.int32)
        
        # Inference (assume max generated len is 64)
        gen_len = 64
        out_ids = infer(params, jnp.array(src_padded), max_len=gen_len, m_v_all=m_v_all, ngram_mem=ngram_mem)
        out_ids = np.array(out_ids)
        
        for b in range(len(batch_texts)):
            # find eos
            eos_id = corpus.get_eos_id()
            seq = out_ids[b].tolist()
            if eos_id in seq:
                seq = seq[:seq.index(eos_id)]
            total_tokens_generated += len(seq)
            predictions.append(corpus.decode(seq))
            
    end_time = time.time()
    
    # Calculate Metrics
    bleu = sacrebleu.corpus_bleu(predictions, [refs])
    chrf = sacrebleu.corpus_chrf(predictions, [refs])
    
    latency = end_time - start_time
    latency_per_sentence = latency / len(src_texts)
    tokens_per_sec = total_tokens_generated / latency if latency > 0 else 0
    
    return {
        "Model": model_name,
        "BLEU": bleu.score,
        "chrF": chrf.score,
        "Tok/s": tokens_per_sec,
        "Latency": latency_per_sentence * 1000 # ms
    }

def main():
    print("=== Evaluasi Pipeline (Menjawab RQ1, RQ2, RQ3) ===")
    
    src_path = "../dataset/OpenSubtitles.en-id.en"
    tgt_path = "../dataset/OpenSubtitles.en-id.id"
    
    corpus = StreamingCorpus(src_path, tgt_path)
    vocab_tgt = corpus.vocab_size()
    
    print("Memuat dataset test (1000 kalimat)...")
    src_texts, tgt_refs = read_test_set(src_path, tgt_path, start_line=74000, end_line=75000)
    
    print("Memuat Checkpoint Model SNN...")
    try:
        ckpt = np.load("best_model_jax.npz")
        params = {k: jnp.array(v) for k, v in ckpt.items()}
    except FileNotFoundError:
        print("Model belum dilatih! Jalankan pipeline.py secara utuh.")
        return
        
    m_v_all = precompute_all_sdr(vocab_tgt, d_d, num_active_sdr)
    
    print("Membangun N-Gram Memory untuk RQ2...")
    ngram = SparseNGramMemory()
    ngram.build_from_corpus(corpus, 74000) # Hanya dari train set
    
    # RQ1: SNN Baseline
    res_snn = evaluate_model("SNN (~1M Params)", params, corpus, src_texts, tgt_refs, m_v_all, None)
    
    # RQ2: SNN + N-Gram
    res_ngram = evaluate_model("SNN + N-Gram", params, corpus, src_texts, tgt_refs, m_v_all, ngram)
    
    # Print Markdown Table for Paper
    print("\n\n" + "="*50)
    print(" HASIL EVALUASI UNTUK PAPER (metodelogi.md)")
    print("="*50)
    print("\n| Model        | BLEU | chrF | Tok/s | Latency (ms) |")
    print("| ------------ | ---: | ---: | ----: | -----------: |")
    for res in [res_snn, res_ngram]:
        print(f"| {res['Model']:<17} | {res['BLEU']:5.2f} | {res['chrF']:5.2f} | {res['Tok/s']:5.1f} | {res['Latency']:5.1f} |")
        
    print("\nKesimpulan RQ:")
    print("RQ1 Terjawab: Lihat baris pertama (SNN) apakah BLEU > 0.")
    print("RQ2 Terjawab: Bandingkan BLEU SNN vs SNN+N-Gram.")
    print("RQ3 Terjawab: Bandingkan Tok/s dan Latency kedua model.")

if __name__ == "__main__":
    main()
