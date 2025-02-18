use anyhow::Result;
use log::info;
use std::{fs::File, io::Read, path::Path};
use tokenizers;

fn load_file(path: &String) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

pub fn main() -> Result<()> {
    // 初始化日志
    env_logger::init();

    // 加载tokenizer
    let tokenizer = tokenizers::Tokenizer::from_file("bpe_tokenizer_magical.json").unwrap();

    // 读取文件内容
    let text = load_file(&"./res/articles/super_magical_emperior.txt".to_string())?;

    // 对文本进行编码
    let encoded = tokenizer.encode(text, true).unwrap();
    let tokens = encoded.get_ids();

    // 获取token数量
    let token_count = tokens.len();

    println!("文件包含 {} 个token", token_count);

    // 输出前10个token用于验证
    if token_count > 0 {
        let preview_count = token_count.min(10);
        println!("\n前{}个token的解码结果:", preview_count);
        for i in 0..preview_count {
            let token_id = tokens[i];
            let decoded = tokenizer.decode(&[token_id], true).unwrap();
            println!("Token {}: {} -> '{}'", i + 1, token_id, decoded);
        }
    }

    Ok(())
}
