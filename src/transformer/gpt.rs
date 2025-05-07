use crate::data::Dataset;

use super::{feed_forward::FeedForward, multi_head::MultiHeadAttention, *};
use candle_core::safetensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, layer_norm, linear, loss::cross_entropy, AdamW, Embedding, LayerNorm,
    LayerNormConfig, Optimizer as _, ParamsAdamW, VarBuilder, VarMap,
};
use candle_transformers::generation::LogitsProcessor;
use log::{error, info};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
use tokenizers::Tokenizer;

pub struct Block {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm, 

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
        })
    }

    pub fn forward_with_cache(&self, x: &Tensor, k_cache: Option<&Tensor>, v_cache: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        let ln1 = self.ln1.forward(x)?;
        // 使用带缓存的注意力机制
        let (attn_output, k_out, v_out) = self.self_attn.forward_with_cache(&ln1, k_cache, v_cache)?;
        let atten_x = attn_output.add(&x)?;
        let x_ln2 = self.ln2.forward(&atten_x)?;
        let ff_output = self.feed_forward.forward(&x_ln2)?;
        let output = ff_output.add(&atten_x)?;
        Ok((output, k_out, v_out))
    }
}

impl Module for Block {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let ln1 = self.ln1.forward(x)?;
        let sa = self.self_attn.forward(&ln1)?;
        let rx = x.add(&sa)?;
        let ln2 = self.ln2.forward(&rx)?;
        let ffw = self.feed_forward.forward(&ln2)?;
        let rx = rx.add(&ffw)?;
        Ok(rx)
    }
}

pub struct GPTModel {
    token_embedding: Embedding,
    blocks: Vec<Block>,
    layer_norm: LayerNorm,
    lm_head: Linear,
    cfg: Config,
    var_map: Option<VarMap>,
    tokenizer: Tokenizer,
    optimizer: Option<AdamW>,
}

impl GPTModel {
    pub fn new(cfg: &Config, device: &Device, tokenizer: Tokenizer) -> Result<Self> {
        let var_map = VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, device);
        Self::build(cfg, vb, Some(var_map), tokenizer)
    }

    pub fn load(cfg: &Config, path: &str, tokenizer: Tokenizer) -> Result<Self> {
        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &cfg.device)? };

        Self::build(cfg, var_builder, None, tokenizer)
    }

    fn build(
        cfg: &Config,
        vb: VarBuilder,
        var_map: Option<VarMap>,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let token_embedding = embedding(cfg.n_vocab, cfg.n_embd, vb.pp("token_embedding"))?;
        let blocks = (0..cfg.n_layer)
            .map(|i| Block::new(vb.pp(&format!("blocks.{}", i)), cfg))
            .collect::<Result<Vec<_>>>()?;
        let ln_f = layer_norm(cfg.n_embd, LayerNormConfig::from(1e-5), vb.pp("ln_f"))?;
        let lm_head = linear(cfg.n_embd, cfg.n_vocab, vb.pp("lm_head"))?;

        let mut optimizer = None;
        if cfg.training {
            let opt = AdamW::new(var_map.as_ref().unwrap().all_vars(), ParamsAdamW::default())?;
            optimizer = Some(opt);
        }


        Ok(Self {
            token_embedding,
            blocks,
            layer_norm: ln_f,
            lm_head,
            cfg: cfg.clone(),
            var_map,
            tokenizer,
            optimizer,
        })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        if let Some(var_map) = &self.var_map {
            let var_map = var_map.data().lock().unwrap();
            let tensors = var_map
                .iter()
                .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
                .collect();
            safetensors::save(&tensors, path)?;
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn train(
        &mut self,
        dataset: &mut Dataset,
        num_epochs: usize,
        batch_size: usize,
        running: &Arc<AtomicBool>,
    ) -> Result<()> {
        info!(
            "var_map parameters size: {:?}",
            self.var_map.as_ref().unwrap().all_vars().len()
        );

        // 获取所有可能的训练窗口
        let total_windows = dataset.get_total_training_windows(self.cfg.n_ctx)?;
        let mut optimizer = self.optimizer.take().unwrap();
        info!("total training windows: {}", total_windows);
        
        if total_windows == 0 {
            error!("no enough training data");
            return Ok(());
        }

        for epoch in 0..num_epochs {
            // 检查是否需要停止训练
            if !running.load(Ordering::SeqCst) {
                info!("receive stop signal, end training");
                break;
            }

            // 跟踪每个epoch的总损失
            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            // 对每个batch进行训练
            for batch_idx in (0..total_windows).step_by(batch_size) {
                // 再次检查是否需要停止训练
                if !running.load(Ordering::SeqCst) {
                    info!("receive stop signal, end training");
                    return Ok(());
                }

                let actual_batch_size = batch_size.min(total_windows - batch_idx);
                
                let (training_inputs, training_targets) = match dataset
                    .get_sequential_training_batch(batch_idx, actual_batch_size, self.cfg.n_ctx)
                {
                    Ok(Some(result)) => result,
                    Ok(None) => {
                        info!("skip this batch");
                        continue;
                    }
                    Err(e) => {
                        error!("error when get training batch: {:?}", e);
                        continue;
                    }
                };
                

                let logits = self.forward(&training_inputs)?;
                let (batch_size, context_size, embedding_size) = logits.shape().dims3()?;
                let logits = logits.reshape((batch_size * context_size, embedding_size))?;
                let targets = training_targets.reshape((batch_size * context_size,))?;
                let loss = cross_entropy(&logits, &targets)?;

                optimizer.backward_step(&loss)?;

                let loss_val = loss.to_scalar::<f32>()?;
                epoch_loss += loss_val;
                batch_count += 1;
                
            }

            // 每个epoch结束后输出平均损失
            if batch_count > 0 {
                let avg_loss = epoch_loss / batch_count as f32;
                info!(
                    "Epoch {}/{} completed. Average loss: {}",
                    epoch + 1,
                    num_epochs,
                    avg_loss
                );

                // 每个epoch结束后保存模型
                // if let Err(e) = self.save("gpt_model_checkpoint.bin") {
                //     error!("Failed to save checkpoint: {:?}", e);
                // } else {
                //     info!("Checkpoint saved after epoch {}", epoch + 1);
                // }
            }
        }
        self.optimizer = Some(optimizer);
        Ok(())
    }

    pub fn generate(&self, input: &str, max_new_tokens: usize, temperature: f64) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(input, true)
            .unwrap()
            .get_ids()
            .to_vec();
        let input_len = tokens.len();
        
        // 初始化KV缓存
        let mut kv_cache: Option<Vec<(Tensor, Tensor)>> = None;
        let mut logits_processor = LogitsProcessor::new(0, Some(temperature), Some(0.7));

        // 首次处理完整输入序列
        let input_tensor = Tensor::new(tokens.as_slice(), &self.cfg.device)?.unsqueeze(0)?;
        let (logits, new_kv_cache) = self.forward_with_cache(&input_tensor, &[], false)?;
        kv_cache = Some(new_kv_cache);
        
        // 获取最后一个token的logits并采样
        let logits = logits.squeeze(0)?;
        let next_token_logits = logits.get(input_len - 1)?;
        let next_token = logits_processor.sample(&next_token_logits)?;
        tokens.push(next_token);
        
        // 检查是否为EOS
        if next_token == 1 { // </s>的token ID是1
            info!("detect EOS, end generation");
            let decoded = self.tokenizer.decode(tokens.as_slice(), true).unwrap();
            return Ok(decoded);
        }
        
        // 继续生成剩余的token
        for _ in 1..max_new_tokens {
            // 只处理最后一个token
            let last_token = Tensor::new(&[tokens[tokens.len() - 1]], &self.cfg.device)?.unsqueeze(0)?;
            
            // 使用缓存进行前向传播
            let (logits, new_kv_cache) = if let Some(cache) = &kv_cache {
                self.forward_with_cache(&last_token, cache, true)?
            } else {
                self.forward_with_cache(&last_token, &[], false)?
            };
            
            // 更新KV缓存
            kv_cache = Some(new_kv_cache);
            
            // 获取logits并采样下一个token
            let logits = logits.squeeze(0)?;
            let next_token_logits = logits.get(0)?;  // 因为只输入了一个token，所以索引为0
            let next_token = logits_processor.sample(&next_token_logits)?;
            tokens.push(next_token);
            
            // 检查是否为EOS
            if next_token == 1 { // </s>的token ID是1
                info!("detect EOS, end generation");
                break;
            }
        }
        
        let decoded = self.tokenizer.decode(tokens.as_slice(), true).unwrap();
        Ok(decoded)
    }

    pub fn generate_no_cache(&self, input: &str, max_new_tokens: usize, temperature: f64) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(input, true)
            .unwrap()
            .get_ids()
            .to_vec();
        let mut generated_tokens = 0usize;
        
        let mut logits_processor = LogitsProcessor::new(0, Some(temperature), Some(0.6));

        for _ in 0..max_new_tokens {
            // 每次都处理完整的token序列
            let input = Tensor::new(tokens.as_slice(), &self.cfg.device)?.unsqueeze(0)?;
            let logits = self.forward(&input)?;
            
            // 只关注最后一个位置的logits
            let logits = logits.squeeze(0)?;
            let next_token_logits = logits.get(tokens.len() - 1)?;
            
            // 采样下一个token
            let next_token = logits_processor.sample(&next_token_logits)?;
            info!("Token: {:?}", next_token);
            tokens.push(next_token);
            generated_tokens += 1;
            
            // 检查是否为EOS
            if next_token == 1 { // </s>的token ID是1
                info!("detect EOS, end generation");
                break;
            }
        }
        
        info!("generated_tokens without cache: {}", generated_tokens);
        let decoded = self.tokenizer.decode(tokens.as_slice(), true).unwrap();
        Ok(decoded)
    }
    
    // 修复带KV缓存的前向传播方法
    fn forward_with_cache(&self, x: &Tensor, kv_cache: &[(Tensor, Tensor)], use_cache: bool) -> Result<(Tensor, Vec<(Tensor, Tensor)>)> {
        let token_embedding = self.token_embedding.forward(x)?;
        let mut x = token_embedding;
        
        let mut new_kv_cache = Vec::with_capacity(self.blocks.len());
        
        for (i, block) in self.blocks.iter().enumerate() {
            // 使用带缓存的前向传播
            let (k_cache, v_cache) = if use_cache && i < kv_cache.len() {
                (Some(&kv_cache[i].0), Some(&kv_cache[i].1))
            } else {
                (None, None)
            };
            
            let (block_output, k_out, v_out) = block.forward_with_cache(&x, k_cache, v_cache)?;
            x = block_output;
            new_kv_cache.push((k_out, v_out));
        }
        
        let x_norm = self.layer_norm.forward(&x)?;
        let logits = self.lm_head.forward(&x_norm)?;
        
        Ok((logits, new_kv_cache))
    }
}

impl Module for GPTModel {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let token_embedding = self.token_embedding.forward(x)?;
        let mut x = token_embedding;
        
        for block in self.blocks.iter() {
            x = block.forward(&x)?;
        }
        
        let x_norm = self.layer_norm.forward(&x)?;
        let logits = self.lm_head.forward(&x_norm)?;
        
        Ok(logits)
    }
}
