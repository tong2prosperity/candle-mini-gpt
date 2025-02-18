use std::{fs::File, io::Read, io::Write};

use anyhow::Result;
use candle_core::{utils, Device, Shape, Tensor};
use candle_mini_gpt::{
    data::Dataset,
    transformer::{gpt::GPTModel, Config},
};
use chrono::Local;
use env_logger::{Builder, WriteStyle};
use log::{error, info};
use tokenizers;

fn load_file(path: &String) -> anyhow::Result<String> {
    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

pub fn main() -> Result<()> {
    Builder::from_default_env()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} [{}] - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S"),
                record.level(),
                record.args()
            )
        })
        .write_style(WriteStyle::Always)
        .filter_level(log::LevelFilter::Debug)
        .init();

    let tokenizer = tokenizers::Tokenizer::from_file("leader_bpe_tokenizer.json").unwrap();
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

    // try load config from file if failed new a config
    let config = if let Ok(config) = Config::load("config.json", device.clone()) {
        config
    } else {
        Config::new(true, device.clone())
    };

    let mut dataset = load_dataset(&tokenizer, &config.device)?;

    let GPT = GPTModel::new(&config, &config.device, tokenizer)?;
    GPT.train(&mut dataset, 4, 8)?;
    GPT.save("gpt_model.bin")?;
    Ok(())
}

fn load_dataset(tokenizer: &tokenizers::Tokenizer, device: &Device) -> Result<Dataset> {
    let text = load_file(&"./res/articles/super_magical_emperior.txt".to_string())?;
    let encoded = tokenizer.encode(text, true).unwrap();

    let data = Tensor::from_slice(encoded.get_ids(), Shape::from(encoded.len()), device).unwrap();
    let dataset = Dataset::new(data, 0.8);
    Ok(dataset)
}
