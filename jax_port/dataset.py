import json
import numpy as np
from tokenizers import Tokenizer

class StreamingCorpus:
    def __init__(self, src_path, tgt_path, tokenizer_path="../bpe_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.src_path = src_path
        self.tgt_path = tgt_path
        
    def vocab_size(self):
        return self.tokenizer.get_vocab_size(with_added_tokens=True)
        
    def get_pad_id(self):
        return self.tokenizer.token_to_id("<PAD>")
        
    def get_eos_id(self):
        return self.tokenizer.token_to_id("<EOS>")
        
    def encode(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[.,!?"\']', '', text)
        encoding = self.tokenizer.encode(text)
        return encoding.ids
        
    def decode(self, ids):
        return self.tokenizer.decode(ids)
        
    def stream_batches(self, batch_size, limit, max_seq_len=128):
        eos_id = self.get_eos_id()
        with open(self.src_path, 'r', encoding='utf-8') as src_file, \
             open(self.tgt_path, 'r', encoding='utf-8') as tgt_file:
             
            lines_read = 0
            sources, targets = [], []
            for src_line, tgt_line in zip(src_file, tgt_file):
                if lines_read >= limit:
                    break
                    
                src_ids = self.encode(src_line) + [eos_id]
                tgt_ids = self.encode(tgt_line) + [eos_id]
                
                # Truncate if too long
                src_ids = src_ids[:max_seq_len]
                tgt_ids = tgt_ids[:max_seq_len]
                
                sources.append(src_ids)
                targets.append(tgt_ids)
                lines_read += 1
                
                if len(sources) == batch_size:
                    # Implement Shape Bucketing (16, 32, 48, 64, 80, 96, 112, 128)
                    # Ini mencegah JAX kompilasi berulang kali tapi tetap hemat komputasi
                    max_len_src = max(len(s) for s in sources)
                    max_len_tgt = max(len(t) for t in targets)
                    batch_max = max(max_len_src, max_len_tgt)
                    
                    bucket_len = max_seq_len
                    for b in [16, 32, 48, 64, 80, 96, 112, 128]:
                        if batch_max <= b:
                            bucket_len = b
                            break
                            
                    pad_id = self.get_pad_id()
                    
                    src_padded = np.array([s + [pad_id]*(bucket_len - len(s)) for s in sources], dtype=np.int32)
                    tgt_padded = np.array([t + [pad_id]*(bucket_len - len(t)) for t in targets], dtype=np.int32)
                    
                    yield src_padded, tgt_padded
                    sources, targets = [], []

class SparseNGramMemory:
    def __init__(self):
        from collections import defaultdict
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        
    def build_from_corpus(self, corpus, max_lines):
        print(f"Membangun Sparse N-Gram Memory dari corpus target (Max {max_lines} baris)...")
        eos_id = corpus.get_eos_id()
        count = 0
        with open(corpus.tgt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= max_lines:
                    break
                tokens = corpus.encode(line) + [eos_id]
                if len(tokens) > 1:
                    for i in range(1, len(tokens)):
                        prev_tok = tokens[i-1]
                        curr_tok = tokens[i]
                        self.bigram_counts[prev_tok][curr_tok] += 1
                count += 1
        print(f"Sparse N-Gram Memory selesai dibangun dari {count} kalimat.")
        
    def get_candidates(self, prev_token):
        if prev_token in self.bigram_counts:
            counts = self.bigram_counts[prev_token]
            total = sum(counts.values())
            candidates = [(tok, cnt / total) for tok, cnt in counts.items()]
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates
        return []
