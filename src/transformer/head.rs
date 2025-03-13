use candle_core::{DType, Device, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use super::Config;

pub struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    scale: f64,
    tril: Tensor, // lower triangular mask
    dropout_p: f64,
    training: bool,
    neg_inf: Tensor,
    head_size: usize,
}

impl Head {
    pub fn new(vb: VarBuilder, cfg: &Config, head_size: usize) -> Result<Self> {
        // Initialize the linear layers without bias
        let key = linear_no_bias(cfg.n_embd, head_size, vb.pp("key"))?;
        let query = linear_no_bias(cfg.n_embd, head_size, vb.pp("query"))?;
        let value = linear_no_bias(cfg.n_embd, head_size, vb.pp("value"))?;

        // Create lower triangular mask
        let tril = Tensor::tril2(cfg.n_ctx, DType::U32, &cfg.device)?;
        let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.to_device(&cfg.device)?;

        Ok(Self {
            key,
            query,
            value,
            scale: (head_size as f64).powf(-0.5),
            tril,
            dropout_p: cfg.dropout as f64,
            training: false,
            neg_inf,
            head_size,
        })
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    
    pub fn get_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let k = self.key.forward(x)?;
        let q = self.query.forward(x)?;
        let v = self.value.forward(x)?;
        Ok((q, k, v))
    }
    
    
    pub fn attention_with_qkv(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (_, seq_len, _) = q.dims3()?;

        let mut weight = ((q.matmul(&k.transpose(D::Minus2, D::Minus1)?))? * self.scale)?;

        // Apply causal mask
        let masked_weight = self
            .tril
            .i((..seq_len, ..seq_len))?
            .broadcast_as(Shape::from(weight.shape()))?
            .where_cond(
                &weight,
                &self.neg_inf.broadcast_as(Shape::from(weight.shape()))?,
            )?;

        // Apply softmax and dropout
        weight = candle_nn::ops::softmax(&masked_weight, D::Minus1)?;

        // Compute output
        let output = weight.matmul(&v)?;
        Ok(output)
    }
}

impl Module for Head {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.get_qkv(x)?;
        self.attention_with_qkv(&q, &k, &v)
    }
}
