use spiking_translation::snn;


fn main() {
    println!("Starting SpikingTranslation...");
    
    // Membatasi penggunaan CPU maksimal (misal: hanya 4 thread) agar sistem tidak hang/100% load
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap_or_default();
    
    let mut corpus = spiking_translation::data::corpus::StreamingCorpus::new("dataset/OpenSubtitles.en-id.en", "dataset/OpenSubtitles.en-id.id");
    // Kita akan membangun vocab dari 500.000 baris pertama saja agar tidak terlalu lama (bisa dinaikkan jika perlu)
    corpus.build_vocab(500000); 
    
    // Pass 2: Ambil 1 batch secara streaming untuk membuktikan irit RAM
    let mut iter = corpus.stream_batches(32, 100000);
    if let Some((src_batch, _tgt_batch)) = iter.next() {
        println!("Streaming Batch berhasil dimuat! Ukuran batch: {}", src_batch.len());
    }
    
    spiking_translation::train::run_training_loop(&corpus);
}
