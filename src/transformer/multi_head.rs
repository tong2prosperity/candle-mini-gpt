use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use super::head::Head;
use super::Config;

pub struct MultiHeadAttention {
    heads: Vec<Head>,
    proj: Linear,
    dropout_p: f64,
    training: bool,
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
        let proj = Linear::new(
            vb.pp("proj").get((cfg.n_embd, cfg.n_embd), "weight")?,
            Some(vb.pp("proj").get(cfg.n_embd, "bias")?),
        );

        Ok(Self {
            heads,
            proj,
            dropout_p: cfg.dropout as f64,
            training: false,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply each head and collect results
        let mut head_outputs = Vec::with_capacity(self.heads.len());
        for head in &self.heads {
            head_outputs.push(head.forward(x)?);
        }

        // Concatenate all head outputs along the last dimension
        let concat = Tensor::cat(&head_outputs, 2)?;
        
        // Apply final projection and dropout
        let out = self.proj.forward(&concat)?;
        let out = if self.training {
            candle_nn::ops::dropout(&out, self.dropout_p as f32)?
        } else {
            out
        };

        Ok(out)
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for head in &mut self.heads {
            head.set_training(training);
        }
    }
}
