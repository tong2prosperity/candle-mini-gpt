pub mod feed_forward;
pub mod gpt;
pub mod head;
pub mod multi_head;

use candle_core::{DType, Device, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::{Linear, VarBuilder};

const BATCH_SIZE: usize = 64;
const BLOCK_SIZE: usize = 256;
const MAX_ITERS: usize = 5000;
const EVAL_INTERVAL: usize = 500;
const LEARNING_RATE: f32 = 3e-4;
const EVAL_ITERS: usize = 200;

const N_VOCAB: usize = 20000;

const N_EMBED: usize = 384;
const N_HEAD: usize = 6;
const N_LAYER: usize = 6;
const DROPOUT: f32 = 0.2;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub block_size: usize,
    pub dropout: f32,
    pub training: bool,
    pub device: Device,
}

impl Config {
    pub fn new(training: bool, device: Device) -> Self {
        Self {
            n_layer: N_LAYER,
            n_vocab: N_VOCAB,
            n_embd: N_EMBED,
            n_head: N_HEAD,
            block_size: BLOCK_SIZE,
            dropout: DROPOUT,
            training: training,
            device,
        }
    }
}
