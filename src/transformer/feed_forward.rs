use super::*;
use candle_nn::{Linear, Activation, Dropout};

pub struct FeedForward {
    linear1: Linear,
    relu: Activation,
    linear2: Linear,
    dropout: Dropout,
    cfg: Config,
}

impl FeedForward {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let linear1 = Linear::new(vb.get((cfg.n_embd, 4 * cfg.n_embd), "linear1")?, None);
        let relu = Activation::Relu;
        let linear2 = Linear::new(vb.get((4 * cfg.n_embd, cfg.n_embd), "linear2")?, None);
        let dropout = Dropout::new(cfg.dropout);

        Ok(Self { linear1, relu, linear2, dropout, cfg: cfg.clone() })
    }
}




impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = self.relu.forward(&x)?;
        let x = self.linear2.forward(&x)?;
        let x = self.dropout.forward(&x, self.cfg.training)?;
        Ok(x)
    }
}