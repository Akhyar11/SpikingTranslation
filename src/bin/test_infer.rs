use spiking_translation::data::corpus::StreamingCorpus;
use spiking_translation::snn::architecture::{SpikingEncoder, STCM, SpikingDecoder};
use spiking_translation::snn::lif::LifParameters;
use spiking_translation::eval::infer;
use serde::{Deserialize, Serialize};
use ndarray::Array2;
use spiking_translation::data::ngram::SparseNGramMemory;
use std::fs::File;

#[derive(Serialize, Deserialize)]
struct SpikingCheckpoint {
    enc_w_e: Array2<f32>,
    enc_w_r: Array2<f32>,
    stcm_w_ce: Array2<f32>,
    stcm_w_cc: Array2<f32>,
    stcm_w_ctx: Array2<f32>,
    stcm_w_self: Array2<f32>,
    dec_w_y: Array2<f32>,
    dec_w_c: Array2<f32>,
    dec_w_r: Array2<f32>,
}

fn main() {
    println!("Memuat vocabulary BPE...");
    let corpus = StreamingCorpus::new("dataset/OpenSubtitles.en-id.en", "dataset/OpenSubtitles.en-id.id");
    
    let mut ngram_memory = SparseNGramMemory::new();
    ngram_memory.build_from_corpus(&corpus, 75000);
    
    println!("Memuat checkpoint 'best_model.json'...");
    let file = match File::open("best_model.json") {
        Ok(f) => f,
        Err(_) => {
            println!("File checkpoint 'best_model.json' belum ada atau tidak ditemukan!");
            return;
        }
    };
    
    let checkpoint: SpikingCheckpoint = serde_json::from_reader(file).expect("Gagal mem-parsing JSON checkpoint!");
    
    let lif_params = LifParameters::new(0.9, 1.0, 1.0);
    let beta_seq = 0.5;
    
    let encoder = SpikingEncoder::new(checkpoint.enc_w_e, checkpoint.enc_w_r, lif_params.clone(), beta_seq);
    let stcm = STCM::new(checkpoint.stcm_w_ce, checkpoint.stcm_w_cc, checkpoint.stcm_w_ctx, checkpoint.stcm_w_self, lif_params.clone(), beta_seq);
    let decoder = SpikingDecoder::new(checkpoint.dec_w_y, checkpoint.dec_w_c, checkpoint.dec_w_r, lif_params, beta_seq);
    
    println!("\n=== Uji Coba Terjemahan SNN (Inference) ===");
    let texts = vec![
        "-=Episode 13=-",
        "People.",
        "That brat.",
    ];
    
    for text in texts {
        let src_indices = corpus.encode(text);
        let output = infer(&encoder, &stcm, &decoder, &src_indices, 20, 5, None);
        
        let out_text = corpus.decode(&output);
        
        println!("En: {}", text);
        println!("IDs: {:?}", output);
        println!("Id: {}", out_text.trim());
        println!("----------------------");
    }
}
