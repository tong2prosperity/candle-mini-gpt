use candle_core::Result;
use tokenizers;

pub fn main() -> Result<()> {
    let tokenizer = tokenizers::Tokenizer::from_file("leader_bpe_tokenizer.json").unwrap();
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("vocab_size: {}", vocab_size);
    // test tokenizer
    let text = "你好，世界！";
    let tokens = tokenizer.encode(text, true).unwrap();
    println!("tokens: {:?}", tokens);

    Ok(())
}
