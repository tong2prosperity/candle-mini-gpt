use std::{fs::File, io::Read};

use anyhow::Result;
use candle_core::{utils, Device, Shape, Tensor};
use candle_mini_gpt::{
    data::Dataset,
    transformer::{gpt::GPTModel, Config},
};
use tokenizers;
use walkdir::WalkDir;
fn load_file(path: &String) -> anyhow::Result<String> {
    let mut file = File::open(path)?;

    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    Ok(contents)
}

fn collect_txt_files(dir_path: &str) -> Vec<String> {
    WalkDir::new(dir_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext.eq_ignore_ascii_case("txt"))
        })
        .map(|e| e.path().to_string_lossy().into_owned())
        .collect()
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

    let config = Config::new(true, device);
    let mut gpt = GPTModel::new(&config, &config.device, tokenizer)?;

    // 获取所有txt文件路径
    let mut files = collect_txt_files("res/articles");
    println!("Found {} files to train on", files.len());

    files.clear();

    files.push("res/articles/duizhang_short_real.txt".to_string());


    // 逐个文件训练
    for (idx, file_path) in files.iter().enumerate() {
        println!("Training on file {}/{}: {}", idx + 1, files.len(), file_path);
        
        // 加载单个文件的数据集
        let mut dataset = load_single_dataset(&file_path, &gpt.tokenizer, &config.device)?;
        
        // 训练2个epochs
        gpt.train(&mut dataset, 4, 10)?;
        
        // 定期保存模型
        if (idx + 1) % 5 == 0 {
            gpt.save(&format!("gpt_model_checkpoint_{}.bin", idx + 1))?;
        }
    }

    // 保存最终模型
    gpt.save("gpt_model_final.bin")?;
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

fn load_single_dataset(file_path: &str, tokenizer: &tokenizers::Tokenizer, device: &Device) -> Result<Dataset> {
    let text = load_file(&file_path.to_string())?;
    let encoded = tokenizer.encode(text, true).unwrap();

    let data = Tensor::from_slice(encoded.get_ids(), Shape::from(encoded.len()), device).unwrap();
    let dataset = Dataset::new(data, 0.8);
    Ok(dataset)
}
