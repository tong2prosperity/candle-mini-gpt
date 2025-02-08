use std::{fs::File, io::Read};

use anyhow::Result;
use candle_core::{utils, Device, Shape, Tensor};
use candle_mini_gpt::{
    data::Dataset,
    transformer::{gpt::GPTModel, Config},
};
use tokenizers;

fn load_file(path: &String) -> anyhow::Result<String> {
    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

pub fn main() -> Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_file("leader_bpe_tokenizer.json").unwrap();
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("vocab_size: {}", vocab_size);

    let device = {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if utils::metal_is_available() {
                Device::new_metal(0)?
            } else {
                Device::Cpu
            }
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            if utils::cuda_is_available() {
                Device::new_cuda(0)?
            } else {
                Device::Cpu
            }
        }
    };

    let config = Config::new(true, device);

    let mut dataset = load_dataset(&tokenizer, &config.device)?;

    let GPT = GPTModel::new(&config, &config.device, tokenizer)?;
    GPT.train(&mut dataset, 1, 4)?;
    Ok(())
}

fn test_tokenizer() {
    let tokenizer = tokenizers::Tokenizer::from_file("leader_bpe_tokenizer.json").unwrap();
    let text = "你好，世界！";
    let tokens = tokenizer.encode(text, true).unwrap();
    println!("tokens: {:?}", tokens);
    let decoded = tokenizer.decode(tokens.get_ids(), true).unwrap();
    println!("decoded: {}", decoded);
}

fn load_dataset(tokenizer: &tokenizers::Tokenizer, device: &Device) -> Result<Dataset> {
    let text = load_file(
        &"/Users/nixi/Project/ML/rust/candle-mini-gpt/res/articles/super_magical_emperior.txt"
            .to_string(),
    )?;
    let encoded = tokenizer.encode(text, true).unwrap();

    let data = Tensor::from_slice(encoded.get_ids(), Shape::from(encoded.len()), device).unwrap();
    let dataset = Dataset::new(data, 0.8);
    Ok(dataset)
}
