# Rust Transformer 

<img src="pics/head.jpg" width="200">

一个用Rust语言实现的Transformer模型，类GPT架构，支持训练和推理功能。
###  **可以训练复杂的模型，但没必要，只训练模型说一句话，就是那句话。**

## 项目概述

轻量级的Transformer实现，使用Rust语言编写，并基于[candle](https://github.com/huggingface/candle)深度学习框架。项目提供了一个完整的GPT模型实现，包括模型训练、文本生成和分词器训练等功能。


## 项目结构

```
src/
├── bin/                    # 可执行文件
│   ├── inference.rs        # 模型推理程序
│   ├── model_trainer.rs    # 模型训练程序
│   └── tokenizer_train.rs  # 分词器训练程序
├── data.rs                 # 数据集处理
├── lib.rs                  # 库入口
├── runner/                 # 运行器
└── transformer/            # Transformer模型实现
    ├── feed_forward.rs     # 前馈神经网络
    ├── gpt.rs              # GPT模型实现
    ├── head.rs             # 注意力头
    ├── mod.rs              # 模块定义
    ├── multi_head.rs       # 多头注意力
    └── rotary_emb.rs       # 旋转位置编码
```

## 核心功能

1. **GPT模型实现**: 基于Transformer架构的语言模型，支持多层多头注意力机制
2. **BPE分词器**: 支持训练自定义分词器
3. **数据处理**: 高效的数据集加载和批处理功能
4. **训练框架**: 包含完整的模型训练循环和优化器实现
5. **文本生成**: 支持使用训练好的模型生成文本

## 使用方法

### 环境要求

- Rust 1.70+
- 支持CUDA或Metal的设备（可选，用于GPU加速）

### 训练分词器

```bash
cargo run --bin tokenizer_train -- --input <你的文本文件> --output mini_bpe.json
```

### 训练模型

```bash
cargo run --bin model_trainer
```

模型训练程序会自动读取`res/articles/pretrain.txt`文件作为训练数据，并保存训练好的模型到`gpt_model.safetensors`。

### 模型推理

```bash
cargo run --bin inference
```

### 配置参数

模型配置保存在`config.json`文件中，主要包含以下参数：

- `n_layer`: Transformer层数
- `n_vocab`: 词汇表大小
- `n_embd`: 嵌入维度
- `n_head`: 注意力头数量
- `n_ctx`: 上下文窗口大小
- `dropout`: Dropout比率
- `max_position_embeddings`: 最大位置编码数量
- `rope_theta`: RoPE (旋转位置编码) 参数

## 技术细节

### Transformer架构

该实现遵循标准的Transformer架构，每个Transformer块包含：

1. 多头自注意力机制
2. 前馈神经网络
3. 层归一化
4. 残差连接

### 优化

- 支持KV缓存以加速推理
- 使用旋转位置编码(RoPE)提高位置感知能力
- 支持CPU和GPU(CUDA/Metal)推理

### 数据集处理

提供了两种数据加载方式：
- 随机采样训练批次
- 连续窗口训练批次

## 示例

训练完成后，你可以使用以下代码生成文本：

```rust
let GPT = GPTModel::load(&config, "./gpt_model.safetensors", tokenizer)?;
let result = GPT.generate("你不拿,", 30, 0.1)?; // 默认带kvcache
println!("result: {}", result);
```

## 开发

本项目使用Rust标准开发工具链。要构建项目：

```bash
cargo build --release
```

运行测试：

```bash
cargo test
```

## 扩展

该项目可以扩展用于：
- 自定义语言模型训练
- 文本生成应用
- 语言处理任务
- 作为学习Transformer架构的教学工具 