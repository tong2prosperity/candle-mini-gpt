use std::path::Path;

use anyhow::Result;
use candle_core::{utils, Device};
use candle_mini_gpt::transformer::{gpt::GPTModel, Config};
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
                Device::new_metal(0)?
                //Device::Cpu
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

    let mut config = if Path::new("./config.json").exists() {
        Config::load("config.json", device)?
    } else {
        Config::new(true, device)
    };

    config.training = false;
    
    let gpt = GPTModel::load(&config, "./gpt_model.safetensors", tokenizer)?;

    // 默认使用KV缓存的方式生成
    let input = "<s>你不拿,";
    let max_tokens = 30;
    let temperature = 0.1;
    
    println!("输入: {}", input);
    
    // 使用带KV缓存的生成方法
    let start = std::time::Instant::now();
    let result = gpt.generate(input, max_tokens, temperature)?;
    let elapsed = start.elapsed();
    println!("带KV缓存生成结果: {}\nelapsed: {:?}", result, elapsed);
    
    // 对比无缓存的生成方法
    let start_no_cache = std::time::Instant::now();
    let result_no_cache = gpt.generate_no_cache(input, max_tokens, temperature)?;
    let elapsed_no_cache = start_no_cache.elapsed();
    println!("无KV缓存生成结果: {}\nelapsed: {:?}", result_no_cache, elapsed_no_cache);

    Ok(())
}
