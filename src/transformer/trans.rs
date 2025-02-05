
use candle_core::{Device, DType, Tensor};

pub struct Transformer {
    pub model: String,
    pub tokenizer: String,
    pub WQ: Tensor,
    pub WK: Tensor,
    pub WV: Tensor,
    pub WO: Tensor,
}

impl Transformer {
    pub fn new(model: String, tokenizer: String) -> Self {
        let device = Device::new_metal(0).unwrap();
        let WQ = Tensor::new(&[1024, 1024], &device).unwrap();
        let WK = Tensor::new(&[1024, 1024], &device).unwrap();
        let WV = Tensor::new(&[1024, 1024], &device).unwrap();
        let WO = Tensor::new(&[1024, 1024], &device).unwrap();
        Self { model, tokenizer, WQ, WK, WV, WO }
    }
}
