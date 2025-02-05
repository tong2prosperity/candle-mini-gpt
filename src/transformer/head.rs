use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::Config;

pub struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    scale: f64,
    tril: Tensor, // lower triangular mask
    dropout_p: f64,
    training: bool,
}

impl Head {
    pub fn new(
        vb: VarBuilder,
        cfg: &Config,
        head_size: usize,
    ) -> Result<Self> {

        // Initialize the linear layers without bias
        let key = Linear::new(
            vb.get((cfg.n_embd, cfg.n_head), "key")?,
            None
        );
        let query = Linear::new(
            vb.get((cfg.n_embd, cfg.n_head), "query")?,
            None
        );
        let value = Linear::new(
            vb.get((cfg.n_embd, cfg.n_head), "value")?,
            None
        );
        
        // Create lower triangular mask
        let tril = Tensor::tril2(
            cfg.block_size,
            DType::F32,
            vb.device()
        )?;

        Ok(Self {
            key,
            query,
            value,
            scale: (head_size as f64).powf(-0.5),
            tril,
            dropout_p: cfg.dropout as f64,
            training: false,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _n_embd) = x.dims3()?;
        
        // Get key, query and value projections
        let k = self.key.forward(x)?;
        let q = self.query.forward(x)?;
        let v = self.value.forward(x)?;

        // Compute attention scores
        let wei = ((q.matmul(&k.transpose(1, 2)?))? * self.scale)?;

        // Apply causal mask
        let mask = self.tril.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;
        let neg_inf = f32::NEG_INFINITY;
        let mask_f32 = mask.eq(0)?.to_dtype(DType::F32)?;
        mask_f32.broadcast_mul(rhs)
        let wei = ((&wei * &(1.0 - &mask_f32)?)? + (&mask_f32 * neg_inf)?)?;

        // Apply softmax and dropout
        let wei = candle_nn::ops::softmax(&wei, -1)?;
        let wei = if self.training {
            candle_nn::ops::dropout(&wei, self.dropout_p as f32)?
        } else {
            wei
        };

        // Compute output
        let output = wei.matmul(&v)?;
        
        Ok(output)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}