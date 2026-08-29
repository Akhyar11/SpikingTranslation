from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Inisialisasi Tokenizer
tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = Whitespace()

# Buat Trainer
trainer = BpeTrainer(
    vocab_size=24000,
    special_tokens=["<PAD>", "<UNK>", "<EOS>"]
)

# Latih tokenizer
files = [
    "dataset/OpenSubtitles.en-id.en",
    "dataset/OpenSubtitles.en-id.id"
]
print("Mulai melatih Tokenizer (BPE) dengan vocab 4000...")
tokenizer.train(files, trainer)

# Simpan
tokenizer.save("bpe_tokenizer.json")
print("Tokenizer berhasil disimpan ke bpe_tokenizer.json")
