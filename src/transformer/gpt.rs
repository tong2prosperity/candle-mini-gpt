use crate::data::Dataset;

use super::{feed_forward::FeedForward, multi_head::MultiHeadAttention, *};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, loss::cross_entropy, AdamW, Embedding, LayerNorm,
    LayerNormConfig, Optimizer as _, ParamsAdamW, VarBuilder, VarMap,
};
use candle_transformers::generation::LogitsProcessor;
use log::{debug, error, info};
use tokenizers::Tokenizer;

pub struct Block {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    cfg: Config,
}

impl Block {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(vb.pp("self_attn"), cfg)?;
        let feed_forward = FeedForward::new(vb.pp("feed_forward"), cfg)?;
        let ln1 = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln1"))?;
        let ln2 = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln2"))?;

        Ok(Self {
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
    tokenizer: Tokenizer,
}

impl GPTModel {
    pub fn new(cfg: &Config, device: &Device, tokenizer: Tokenizer) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let token_embedding = embedding(cfg.n_vocab, cfg.n_embd, vb.pp("token_embedding"))?;
        let position_embedding = embedding(cfg.n_ctx, cfg.n_embd, vb.pp("position_embedding"))?;
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
            tokenizer,
            lm_head,
        })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.var_map.save(path)?;
        Ok(())
    }

    pub fn load(cfg: &Config, path: &str, tokenizer: Tokenizer) -> Result<Self> {
        let mut var_map = VarMap::new();
        var_map.load(path)?;
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &cfg.device);
        let token_embedding = embedding(cfg.n_vocab, cfg.n_embd, vb.pp("token_embedding"))?;
        let position_embedding = embedding(cfg.n_ctx, cfg.n_embd, vb.pp("position_embedding"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(vb.pp(&format!("blocks.{}", i)), cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln_f"))?;
        let lm_head = linear(cfg.n_embd, cfg.n_vocab, vb.pp("lm_head"))?;

        let model = Self {
            token_embedding,
            position_embedding,
            blocks,
            layer_norm: ln_f,
            cfg: cfg.clone(),
            var_map,
            tokenizer,
            lm_head,
        };
        Ok(model)
    }

    pub fn train(&self, dataset: &mut Dataset, num_epochs: usize, batch_size: usize) -> Result<()> {
        info!(
            "var_map parameters size: {:?}",
            self.var_map.all_vars().len()
        );
        let mut optimizer = AdamW::new(self.var_map.all_vars(), ParamsAdamW::default())?;

        for epoch in 0..num_epochs {
            // 获取所有可能的训练窗口
            let total_windows = dataset.get_total_training_windows(self.cfg.n_ctx)?;

            // 对每个batch进行训练
            for batch_idx in (0..total_windows).step_by(batch_size) {
                let actual_batch_size = batch_size.min(total_windows - batch_idx);
                let (training_inputs, training_targets) = match dataset
                    .get_sequential_training_batch(batch_idx, actual_batch_size, self.cfg.n_ctx)
                {
                    Ok(result) => result,
                    Err(e) => {
                        error!("Error getting sequential training batch: {:?}", e);
                        continue;
                    }
                };

                let logits = self.forward(&training_inputs)?;
                let (batch_size, context_size, embedding_size) = logits.shape().dims3()?;
                let logits = logits.reshape((batch_size * context_size, embedding_size))?;
                let targets = training_targets.reshape((batch_size * context_size,))?;
                let loss = cross_entropy(&logits, &targets)?;

                optimizer.backward_step(&loss)?;

                if batch_idx % 100 == 0 {
                    // also print the current time
                    let current_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
                    info!(
                        "Time: {} Epoch {} Batch {}/{} Loss: {}",
                        current_time,
                        epoch,
                        batch_idx,
                        total_windows,
                        loss.to_scalar::<f32>()?
                    );
                }
            }
        }
        Ok(())
    }

    pub fn generate(&self, input: &str, max_new_tokens: usize, temperature: f64) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(input, true)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut generated_tokens = 0usize;

        let mut logits_processor = LogitsProcessor::new(0, Some(temperature), Some(0.6));

        for _ in 0..max_new_tokens {
            // cap the tokens to context size
            let token_len = tokens.len();
            if token_len > self.cfg.n_ctx {
                tokens = tokens
                    .into_iter()
                    .skip(token_len - self.cfg.n_ctx)
                    .collect();
            }
            let input = Tensor::new(tokens.as_slice(), &self.cfg.device)?.unsqueeze(0)?;
            // temperature sampling
            let logits = self.forward(&input)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(token_len - 1)?;

            let next_token = logits_processor.sample(&logits)?;
            debug!("next_token: {:?}", next_token);
            tokens.push(next_token);
            generated_tokens += 1;
        }
        info!("generated_tokens: {}", generated_tokens);
        let decoded = self.tokenizer.decode(tokens.as_slice(), true).unwrap();
        Ok(decoded)
    }
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
