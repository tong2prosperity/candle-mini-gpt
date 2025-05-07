use std::{
    fs::File,
    io::Read,
    io::Write,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    sync::Arc,
};

use anyhow::Result;
use candle_core::{utils, Device, Shape, Tensor};
use candle_mini_gpt::{
    data::Dataset,
    transformer::{gpt::GPTModel, Config},
};
use chrono::Local;
use ctrlc;
use env_logger::{Builder, WriteStyle};
use log::{debug, error, info};
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
        //.filter_level(log::LevelFilter::Info)
        .init();

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        info!("收到停止信号,准备保存模型并退出...");
        r.store(false, Ordering::SeqCst);
    })?;

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

    // try load config from file if failed new a config
    let mut config = if let Ok(config) = Config::load("config.json", device.clone()) {
        config
    } else {
        Config::new(true, device.clone())
    };

    config.training = true;

    let mut dataset = load_dataset(&tokenizer, &config.device)?;

    let gpt = GPTModel::new(&config, &config.device, tokenizer)?;

    train_model(gpt, &mut dataset, &config, running)?;

    Ok(())
}

fn train_model(
    mut gpt: GPTModel,
    dataset: &mut Dataset,
    config: &Config,
    running: Arc<AtomicBool>,
) -> Result<()> {
    info!("开始训练模型...");

    while running.load(Ordering::SeqCst) {
        match gpt.train(dataset, 10000, 1, &running) {
            Ok(_) => {
                info!("训练完成一个周期");
                config.save(&Path::new("config.json"))?;
                gpt.save("gpt_model.safetensors")?;
                break;
            }
            Err(e) => {
                error!("训练出错: {}", e);
                break;
            }
        }
    }

    info!("保存模型和配置...");
    config.save(&Path::new("config.json"))?;
    gpt.save("gpt_model.safetensors")?;

    info!("训练结束");
    Ok(())
}

fn load_dataset(tokenizer: &tokenizers::Tokenizer, device: &Device) -> Result<Dataset> {
    let text = load_file(&"./res/articles/pretrain.txt".to_string())?;
    let encoded = tokenizer.encode(text, true).unwrap();
    debug!("after encoding dataset is {:?}", encoded.get_ids());

    let data = Tensor::from_slice(encoded.get_ids(), Shape::from(encoded.len()), device).unwrap();
    let dataset = Dataset::new(data, 1.0);
    Ok(dataset)
}
