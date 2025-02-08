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
}

impl Head {
    pub fn new(vb: VarBuilder, cfg: &Config, head_size: usize) -> Result<Self> {
        // Initialize the linear layers without bias
        let key = linear_no_bias(cfg.n_embd, head_size, vb.pp("key"))?;
        let query = linear_no_bias(cfg.n_embd, head_size, vb.pp("query"))?;
        let value = linear_no_bias(cfg.n_embd, head_size, vb.pp("value"))?;

        // Create lower triangular mask
        let tril = Tensor::tril2(cfg.n_ctx, DType::U32, vb.device())?;

        let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.to_device(vb.device())?;

        Ok(Self {
            key,
            query,
            value,
            scale: (head_size as f64).powf(-0.5),
            tril,
            dropout_p: cfg.dropout as f64,
            training: false,
            neg_inf,
        })
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Module for Head {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_, seq_len, n_embed) = x.dims3()?;

        // Get key, query and value projections
        let k = self.key.forward(x)?;
        let q = self.query.forward(x)?;
        let v = self.value.forward(x)?;

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
        weight = candle_nn::ops::dropout(&weight, self.dropout_p as f32)?;

        // Compute output
        let v = self.value.forward(&x)?;
        let output = weight.matmul(&v)?;
        Ok(output)
    }
}
