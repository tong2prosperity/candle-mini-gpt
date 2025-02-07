use candle_core::{Device, Result, Tensor};
use candle_mini_gpt::hello_ml::HelloML;

fn main() -> Result<()> {
    let device = Device::new_metal(0)?;
    let first = Tensor::randn(0f32, 1.0, (784, 100), &device)?;
    let second = Tensor::randn(0f32, 1.0, (100, 10), &device)?;
    let model = HelloML { first, second };

    let dummy_input = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_input)?;

    println!("digit: {:?}", digit);
    Ok(())
}

#[cfg(test)]
mod tests {
    use tokenizers::{models::bpe::BPE, Tokenizer};

    use super::*;

    #[test]
    fn test_tokenizer() {
        let tokenizer = Tokenizer::from_file("leader_bpe_tokenizer.json").unwrap();
        let text = "我爱你";
        let tokens = tokenizer.encode(text, true).unwrap();
        println!("tokens: {:?}", tokens);
    }
}
