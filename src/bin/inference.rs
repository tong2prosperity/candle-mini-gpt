use std::{fs::File, io::Read, path::Path};

use anyhow::Result;
use candle_core::{utils, Device, Shape, Tensor};
use candle_mini_gpt::{
    data::Dataset,
    transformer::{gpt::GPTModel, Config},
};
use tokenizers;

pub fn main() -> Result<()> {
    env_logger::init();
    let tokenizer = tokenizers::Tokenizer::from_file("mini_bpe.json").unwrap();
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("vocab_size: {}", vocab_size);

    let device = {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if utils::metal_is_available() {
                //Device::new_metal(0)?
                Device::Cpu
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

    let config = if Path::new("./config.json").exists() {
        Config::load("config.json", device)?
    } else {
        Config::new(true, device)
    };

    let GPT = GPTModel::load(&config, "./gpt_model.safetensors", tokenizer)?;

    let result = GPT.generate("你不拿", 5, 0.1)?;
    println!("result: {}", result);

    Ok(())
}
