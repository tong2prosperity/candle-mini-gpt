use super::*;
use candle_nn::{linear, Activation, Dropout, Linear};

pub struct FeedForward {
    linear1: Linear,
    relu: Activation,
    linear2: Linear,
    dropout: Dropout,
    cfg: Config,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let linear1 = linear(cfg.n_embd, 4 * cfg.n_embd, vb.pp("linear1"))?;
        let relu = Activation::Gelu;
        let linear2 = linear(4 * cfg.n_embd, cfg.n_embd, vb.pp("linear2"))?;
        let dropout = Dropout::new(cfg.dropout);

        Ok(Self {
            linear1,
            relu,
            linear2,
            dropout,
            cfg: cfg.clone(),
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.relu.forward(&x)?;
        let x = self.linear2.forward(&x)?;
        let x = self.dropout.forward(&x, self.cfg.training)?;
        Ok(x)
    }
}
