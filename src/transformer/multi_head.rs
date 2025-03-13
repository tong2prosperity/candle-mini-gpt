use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use super::head::Head;
use super::Config;
use super::rotary_emb::RotaryEmbedding;

pub struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout_p: f64,
    training: bool,
    rotary_emb: Option<RotaryEmbedding>,
    head_size: usize,
}

impl MultiHeadAttention {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_size = cfg.n_embd / cfg.n_head;
        // Create multiple heads
        let mut heads = Vec::with_capacity(cfg.n_head);
        for i in 0..cfg.n_head {
            let head_vb = vb.pp(&format!("head_{}", i));
            heads.push(Head::new(head_vb, cfg, head_size)?);
        }
        // Final projection layer
        let proj = linear(cfg.n_embd, cfg.n_embd, vb.pp("proj"))?;
        
        // 创建旋转位置编码
        let rotary_emb = Some(RotaryEmbedding::new(DType::F32, cfg, &cfg.device)?);
        
        Ok(Self {
            heads,
            proj,
            dropout_p: cfg.dropout as f64,
            training: false,
            rotary_emb,
            head_size,
        })
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for head in &mut self.heads {
            head.set_training(training);
        }
    }

    // 添加带KV缓存的前向传播方法
    pub fn forward_with_cache(&self, x: &Tensor, k_cache: Option<&Tensor>, v_cache: Option<&Tensor>) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // 收集所有头的q、k、v
        let mut all_q = Vec::with_capacity(self.heads.len());
        let mut all_k = Vec::with_capacity(self.heads.len());
        let mut all_v = Vec::with_capacity(self.heads.len());
        
        for head in &self.heads {
            let (q, k, v) = head.get_qkv(x)?;
            all_q.push(q);
            all_k.push(k);
            all_v.push(v);
        }
        
        // 将所有头的q、k、v合并为一个张量
        let q = Tensor::cat(&all_q, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?; // [batch, n_head, seq_len, head_size]
        let k = Tensor::cat(&all_k, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?;
        let v = Tensor::cat(&all_v, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?;
        
        // 合并缓存的k和v（如果有）
        let (k_combined, v_combined) = if let (Some(k_prev), Some(v_prev)) = (k_cache, v_cache) {
            (Tensor::cat(&[k_prev, &k], 2)?, Tensor::cat(&[v_prev, &v], 2)?)
        } else {
            (k.clone(), v.clone())
        };
        
        // 应用旋转位置编码
        let (q_rot, k_rot) = if let Some(rotary) = &self.rotary_emb {
            let offset = if k_cache.is_some() {
                k_cache.unwrap().dim(2)? // 使用缓存的序列长度作为偏移量
            } else {
                0
            };
            rotary.apply_rotary_emb_qkv(&q, &k_combined, offset)?
        } else {
            (q, k_combined.clone())
        };
        
        // 计算注意力
        let mut head_outputs = Vec::with_capacity(self.heads.len());
        for i in 0..self.heads.len() {
            let q_head = q_rot.narrow(1, i, 1)?;
            let k_head = k_rot.narrow(1, i, 1)?;
            let v_head = v_combined.narrow(1, i, 1)?;
            
            let output = self.heads[i].attention_with_qkv(&q_head, &k_head, &v_head)?;
            head_outputs.push(output);
        }
        
        // 合并所有头的输出
        let concat = Tensor::cat(&head_outputs, D::Minus1)?;
        
        // 应用最终投影
        let out = self.proj.forward(&concat)?;
        
        Ok((out, k_combined, v_combined))
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        
        // 收集所有头的q、k、v
        let mut all_q = Vec::with_capacity(self.heads.len());
        let mut all_k = Vec::with_capacity(self.heads.len());
        let mut all_v = Vec::with_capacity(self.heads.len());
        
        for head in &self.heads {
            let (q, k, v) = head.get_qkv(x)?;
            all_q.push(q);
            all_k.push(k);
            all_v.push(v);
        }
        
        // 将所有头的q、k、v合并为一个张量
        let q = Tensor::cat(&all_q, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?; // [batch, n_head, seq_len, head_size]
        let k = Tensor::cat(&all_k, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?;
        let v = Tensor::cat(&all_v, D::Minus1)?.reshape((batch_size, seq_len, self.heads.len(), self.head_size))?
            .permute((0, 2, 1, 3))?;
        
        // 应用旋转位置编码
        let (q_rot, k_rot) = if let Some(rotary) = &self.rotary_emb {
            rotary.apply_rotary_emb_qkv(&q, &k, 0)?
        } else {
            (q, k)
        };
        
        // 计算注意力
        let mut head_outputs = Vec::with_capacity(self.heads.len());
        for i in 0..self.heads.len() {
            let q_head = q_rot.narrow(1, i, 1)?;
            let k_head = k_rot.narrow(1, i, 1)?;
            let v_head = v.narrow(1, i, 1)?;
            
            let output = self.heads[i].attention_with_qkv(&q_head, &k_head, &v_head)?;
            head_outputs.push(output);
        }
        
        // 合并所有头的输出
        let concat = Tensor::cat(&head_outputs, D::Minus1)?;
        
        // 应用最终投影
        let out = self.proj.forward(&concat)?;
        
        Ok(out)
    }
}
