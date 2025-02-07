use crate::data::Dataset;

use super::{feed_forward::FeedForward, multi_head::MultiHeadAttention, *};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, loss::cross_entropy, AdamW, Embedding, LayerNorm,
    LayerNormConfig, Optimizer as _, ParamsAdamW, VarBuilder, VarMap,
};

pub struct Block {
    var_map: VarMap,
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    cfg: Config,
}

impl Block {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &cfg.device);
        let self_attn = MultiHeadAttention::new(vb.clone(), cfg)?;
        let feed_forward = FeedForward::new(vb.clone(), cfg)?;
        let ln1 = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln2"))?;

        Ok(Self {
            var_map,
            self_attn,
            feed_forward,
            ln1,
            ln2,
            cfg: cfg.clone(),
        })
    }
}

impl Module for Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rx = self.ln1.forward(x)?;
        let rx = self.self_attn.forward(&rx)?;
        let rx = rx.add(x)?;
        let rx = self.ln2.forward(&rx)?;
        let rx = self.feed_forward.forward(&rx)?;
        let rx = rx.add(x)?;
        Ok(rx)
    }
}

pub struct GPTModel {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    layer_norm: LayerNorm,
    lm_head: Linear,
    cfg: Config,
    var_map: VarMap,
}

impl GPTModel {
    pub fn new(cfg: &Config, device: &Device) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let token_embedding = embedding(cfg.n_vocab, cfg.n_embd, vb.pp("token_embedding"))?;
        let position_embedding =
            embedding(cfg.block_size, cfg.n_embd, vb.pp("position_embedding"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(vb.pp(&format!("blocks.{}", i)), cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln_f"))?;

        let lm_head = linear(cfg.n_embd, cfg.n_vocab, vb.pp("lm_head"))?;

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            layer_norm: ln_f,
            cfg: cfg.clone(),
            var_map,
            lm_head,
        })
    }

    pub fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            let (training_inputs, training_targets) =
                dataset.random_training_batch(self.cfg.block_size, batch_size)?;

            let logits = self.forward(&training_inputs)?;
            let (batch_size, context_size, embedding_size) = logits.shape().dims3()?;
            let logits = logits.reshape((batch_size * context_size, embedding_size))?;
            let targets = training_targets.reshape((batch_size * context_size,))?;
            let loss = cross_entropy(&logits, &targets)?;

            optimizer.backward_step(&loss)?;

            println!("Epoch {} Training loss: {}", epoch, loss);
        }

        Ok(())
    }

    // fn generate(&self, x: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
    //     let mut x = x.clone();
    //     for _ in 0..max_new_tokens {
    //         let logits = self.forward(&x)?;
    //         let next_token = logits.argmax(D::Minus1)?;
    //     }
    // }
}

impl Module for GPTModel {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape_of_x = x.shape();
        let token_embedding = self.token_embedding.forward(x)?;
        let context_length = shape_of_x.dims2()?;
        let pos_tensor = Tensor::arange(0, context_length.1 as u32, x.device())?;
        let position_embedding = self.position_embedding.forward(&pos_tensor)?;
        let mut x = token_embedding.broadcast_add(&position_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        let x_norm = self.layer_norm.forward(&x)?;

        let logits = self.lm_head.forward(&x_norm)?;
        Ok(logits)
    }
}
