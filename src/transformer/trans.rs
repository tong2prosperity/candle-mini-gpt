use super::{feed_forward::FeedForward, multi_head::MultiHeadAttention, *};
use candle_core::{Device, DType, Tensor};
use candle_nn::{layer_norm, LayerNorm, LayerNormConfig};

pub struct Block {
    self_attn: MultiHeadAttention,
    feed_forward: FeedForward,
    ln1: LayerNorm,
    ln2: LayerNorm,
    cfg: Config,
}

impl Block {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(vb.clone(), cfg)?;
        let feed_forward = FeedForward::new(vb.clone(), cfg)?;
        let ln1 = layer_norm(
            cfg.n_embd,
            LayerNormConfig::from(1e-5),
            vb.pp("ln1")  
        )?;
        let ln2 = layer_norm(
            cfg.n_embd,
            LayerNormConfig::from(1e-5),
            vb.pp("ln2")  
        )?;

        Ok(Self { self_attn, feed_forward, ln1, ln2, cfg: cfg.clone() })
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