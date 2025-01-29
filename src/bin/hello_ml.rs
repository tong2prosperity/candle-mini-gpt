use candle_core::{Device, Result, Tensor};
use candle_transformer::hello_ml::HelloML;

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let first = Tensor::randn(0f32, 1.0, (784,100), &device)?;
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let model = HelloML { first, second };

    let dummy_input = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit  = model.forward(&dummy_input)?;

    println!("digit: {:?}", digit);
    Ok(())
}