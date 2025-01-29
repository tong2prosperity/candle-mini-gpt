use candle_core::{Device, Result, Tensor};

pub struct HelloML {
    pub first: Tensor,
    pub second: Tensor,
}

impl HelloML {
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let x = image.matmul(&self.first)?;
        let x = x.relu()?;
        x.matmul(&self.second)
    }
}
