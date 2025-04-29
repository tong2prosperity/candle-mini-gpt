use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tokenizers::models::bpe::{BpeTrainer, BpeTrainerBuilder, BPE};
use tokenizers::normalizers::{Sequence, Strip, NFC, NFKC};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::delimiter::CharDelimiterSplit;
use tokenizers::tokenizer::{Tokenizer, TokenizerBuilder};
use tokenizers::AddedToken;

use std::fs;
use walkdir::WalkDir;

fn count_chinese_chars_detailed(
    files: &[String],
) -> Result<(usize, HashMap<char, usize>), std::io::Error> {
    let mut unique_chars = HashSet::new();
    let mut char_frequency = HashMap::new();
    let mut total_chars = 0;

    for file_path in files {
        let content = fs::read_to_string(file_path)?;
        for c in content.chars() {
            if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
                unique_chars.insert(c);
                *char_frequency.entry(c).or_insert(0) += 1;
                total_chars += 1;
            }
        }
    }

    // 按频率排序
    let mut freq_vec: Vec<_> = char_frequency.iter().collect();
    freq_vec.sort_by(|a, b| b.1.cmp(a.1));

    println!("Total Chinese characters: {}", total_chars);
    println!("Unique Chinese characters: {}", unique_chars.len());
    println!("\nTop 10 most frequent characters:");
    for (char, freq) in freq_vec.iter().take(10) {
        println!("{}: {} times", char, freq);
    }

    Ok((unique_chars.len(), char_frequency))
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

#[cfg(test)]
mod test {
    use std::fs::File;

    use super::*;
    #[test]
    fn test_count_chinese_chars_detailed() {
        let files = collect_txt_files("res/articles");
        let (unique_chars, char_frequency) = count_chinese_chars_detailed(&files).unwrap();
        println!("unique_chars: {:?}", unique_chars);
        println!("char_frequency: {:?}", char_frequency);
    }

    #[test]
    fn test_translate() {
        use opencc_rust::*;

        let opencc = OpenCC::new(DefaultConfig::TW2SP).unwrap();

        // 读取文件
        let file_path = "res/articles/队长短文集 .txt";

        // 读取繁体内容
        let content = fs::read_to_string(file_path).unwrap();

        // 转换为简体
        let simplified = opencc.convert(&content);

        // 输出到新文件（添加.simplified后缀）
        let path = Path::new(file_path);
        let new_path = path.with_extension("simplified.txt");
        fs::write(&new_path, simplified).unwrap();

        println!("Converted {} to {}", path.display(), new_path.display());
    }
}

fn main() -> Result<()> {
    // 初始化 BPE trainer
    let mut trainer = BpeTrainerBuilder::new()
        .show_progress(true)
        .vocab_size(22)
        .min_frequency(2)
        .special_tokens(vec![
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("</s>"), true),
            AddedToken::from(String::from("<unk>"), true),
        ])
        .build();
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(Some(Sequence::new(vec![
            Strip::new(true, true).into(),
            NFKC.into(),
        ])))
        //.with_pre_tokenizer(Some(ByteLevel::default()))
        .with_pre_tokenizer(Some(CharDelimiterSplit::new('\u{0}')))
        .with_post_processor(Some(ByteLevel::default()))
        .with_decoder(Some(ByteLevel::default()))
        .build()
        .unwrap();
    // 创建 tokenizer
    // let mut tokenizer = TokenizerBuilder::new()
    //     .with_model(BPE::default())
    //     // 使用 NFKC 归一化，并移除空格
    //     .with_normalizer(Some(Sequence::new(vec![
    //         NFKC.into(),
    //         Strip::new(true, true).into(),
    //     ])))
    //     // 使用字符级别的分割
    //     .with_pre_tokenizer(Some(CharDelimiterSplit::new('\u{0}')))
    //     .build().unwrap();

    // 训练文件路径
    let mut files = collect_txt_files("res/articles");

    files.clear();
    files.push("res/articles/tokenizer_train.txt".to_string());
    println!("files: {:?}", files);

    // 训练模型
    tokenizer.train_from_files(&mut trainer, files).unwrap();

    // 保存训练好的 tokenizer
    tokenizer.save("mini_bpe.json", true).unwrap();

    Ok(())
}
