use candle_core::{DType, IndexOp, Module, Result, Shape, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use log::debug;

use super::Config;

pub struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    scale: f64,
    tril: Tensor, // lower triangular mask
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
        let tril = Tensor::tril2(cfg.n_ctx, DType::U32, &cfg.device)?;
        let neg_inf = Tensor::try_from(f32::NEG_INFINITY)?.to_device(&cfg.device)?;

        Ok(Self {
            key,
            query,
            value,
            scale: (head_size as f64).powf(-0.5),
            tril,
            training: false,
            neg_inf,
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
        // 确保张量是连续的
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        
        // 获取形状信息
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();
        
        debug!("q shape: {:?}, k shape: {:?}, v shape: {:?}", q_shape, k_shape, v_shape);
        
        // 获取维度
        let (batch_size, n_heads, seq_len, head_size) = q.dims4()?;
        let k_seq_len = k.dim(2)?;
        
        // 重塑张量以便进行矩阵乘法
        let q_2d = q.reshape((batch_size * n_heads, seq_len, head_size))?;
        let k_2d = k.reshape((batch_size * n_heads, k_seq_len, head_size))?;
        let v_2d = v.reshape((batch_size * n_heads, v.dim(2)?, v.dim(3)?))?;
        
        // 计算注意力分数
        let mut weight = (q_2d.matmul(&k_2d.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;
        
        // 获取实际的序列长度
        let actual_seq_len = weight.dim(1)?;
        let actual_k_seq_len = weight.dim(2)?;
        

        if actual_seq_len == 1 {
            weight = candle_nn::ops::softmax(&weight, D::Minus1)?;
        } else if actual_seq_len <= self.tril.dim(0)? {
            let masked_weight = self
                .tril
                .i((..actual_seq_len, ..actual_seq_len))?
                .broadcast_as(Shape::from(weight.shape()))?
                .where_cond(
                    &weight,
                    &self.neg_inf.broadcast_as(Shape::from(weight.shape()))?,
                )?;
            weight = candle_nn::ops::softmax(&masked_weight, D::Minus1)?;
        } else {
            // 如果序列长度超过了预定义的tril大小，我们需要创建一个新的掩码
            let new_tril = Tensor::tril2(actual_k_seq_len, DType::U32, weight.device())?;
            let masked_weight = new_tril
                .broadcast_as(Shape::from(weight.shape()))?
                .where_cond(
                    &weight,
                    &self.neg_inf.broadcast_as(Shape::from(weight.shape()))?,
                )?;
            weight = candle_nn::ops::softmax(&masked_weight, D::Minus1)?;
        }
        
        // 计算输出
        let output = weight.matmul(&v_2d)?;
        
        // 重塑回原始形状
        let output = output.reshape((batch_size, n_heads, seq_len, v.dim(3)?))?;
        
        Ok(output)
    }
}

impl Module for Head {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (q, k, v) = self.get_qkv(x)?;
        self.attention_with_qkv(&q, &k, &v)
    }
}
