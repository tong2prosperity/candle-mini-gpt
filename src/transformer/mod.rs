pub mod feed_forward;
pub mod gpt;
pub mod head;
pub mod multi_head;
pub mod rotary_emb;

use std::{fs::File, path};

use candle_core::{DType, Device, IndexOp, Module, Shape, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use serde::{Deserialize, Serialize};

use anyhow::Result;
const MAX_ITERS: usize = 1000;

const BATCH_SIZE: usize = 12;
const CONTEXT_SIZE: usize = 64;

const LEARNING_RATE: f32 = 3e-4;
const N_VOCAB: usize = 22;
const N_EMBED: usize = 32;
const N_HEAD: usize = 4;
const N_LAYER: usize = 4;
const DROPOUT: f32 = 0.1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSerde {
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ctx: usize,
    pub dropout: f32,
    pub training: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,
    pub n_vocab: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_ctx: usize,
    pub dropout: f32,
    pub training: bool,
    pub device: Device,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
}

impl Config {
    pub fn new(training: bool, device: Device) -> Self {
        Self {
            n_layer: N_LAYER,
            n_vocab: N_VOCAB,
            n_embd: N_EMBED,
            n_head: N_HEAD,
            n_ctx: CONTEXT_SIZE,
            dropout: DROPOUT,
            training: training,
            device,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
        }
    }

    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let file = File::open(path)?;
        let config: ConfigSerde = serde_json::from_reader(file)?;

        Ok(Self {
            n_layer: config.n_layer,
            n_vocab: config.n_vocab,
            n_embd: config.n_embd,
            n_head: config.n_head,
            n_ctx: config.n_ctx,
            dropout: config.dropout,
            training: false,
            device: dev,
            max_position_embeddings: config.max_position_embeddings,
            rope_theta: config.rope_theta,
        })
    }

    pub fn save(&self, path: &path::Path) -> Result<()> {
        let file = File::create(path)?;
        let config = ConfigSerde {
            n_layer: self.n_layer,
            n_vocab: self.n_vocab,
            n_embd: self.n_embd,
            n_head: self.n_head,
            n_ctx: self.n_ctx,
            dropout: self.dropout,
            training: self.training,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
        };
        serde_json::to_writer(file, &config)?;
        Ok(())
    }
}
