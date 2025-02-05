pub mod trans;
pub mod head;
pub mod multi_head;


const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const MAX_ITERS: usize = 5000;
const EVAL_INTERVAL: usize = 500;
const LEARNING_RATE: f32 = 3e-4;
const EVAL_ITERS: usize = 200;

const N_EMBED: usize = 384;
const N_HEAD: usize = 6;
const N_LAYER: usize = 6;
const DROPOUT: f32 = 0.2;



pub struct Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub dropout: f32,
}


impl Config {
    pub fn new() -> Self {
        Self { n_embd: N_EMBED, n_head: N_HEAD, block_size: BLOCK_SIZE, dropout: DROPOUT }
    }
}