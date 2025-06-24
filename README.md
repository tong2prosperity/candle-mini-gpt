# 🦀 Rust Transformer - 基于 Candle 的 GPT 实现

这是一个使用 Rust 语言和 Candle 深度学习框架实现的 Decoder-only Transformer 模型（类似 GPT 架构）。项目完整实现了从数据预处理、分词器训练、模型训练到推理的全流程。

## 📖 项目概述

本项目旨在提供一个简洁、高效的 Transformer 实现，具备以下特点：

- **🚀 高性能**：基于 Candle 框架，支持 GPU 加速（CUDA/Metal）
- **🔧 完整流程**：包含分词器训练、模型训练、推理等完整工具链
- **📚 教育友好**：代码结构清晰，适合学习 Transformer 架构
- **🎯 轻量级**：专注核心功能，代码量适中，易于理解和修改

## 🏗️ 项目架构

### 核心模块结构

```
src/
├── lib.rs              # 库入口
├── data.rs             # 数据处理模块
├── transformer/        # Transformer 模型核心
│   ├── mod.rs          # 模块配置和常量
│   ├── gpt.rs          # GPT 主模型实现
│   ├── multi_head.rs   # 多头注意力机制
│   ├── head.rs         # 单个注意力头
│   ├── feed_forward.rs # 前馈神经网络
│   └── rotary_emb.rs   # 旋转位置编码
└── bin/                # 可执行程序
    ├── tokenizer_train.rs  # 分词器训练
    ├── model_trainer.rs    # 模型训练
    └── inference.rs        # 推理程序
```

## 🔍 深入了解各模块

### 1. 数据处理模块 (`data.rs`)

数据处理模块负责：
- **数据集管理**：将原始文本数据转换为训练和验证集
- **批次生成**：支持随机批次和连续批次两种模式
- **窗口滑动**：实现滑动窗口机制，充分利用训练数据

核心功能：
```rust
pub struct Dataset {
    pub training_data: Tensor,     // 训练数据张量
    pub validation_data: Tensor,   // 验证数据张量
    // ... 其他字段
}
```

### 2. Transformer 核心架构

#### 2.1 配置管理 (`mod.rs`)

定义了模型的超参数和配置：
- **模型维度**：嵌入维度、头数、层数等
- **训练参数**：学习率、dropout、上下文长度等
- **硬件配置**：设备选择、数据类型等

#### 2.2 GPT 主模型 (`gpt.rs`)

这是整个项目的核心，实现了完整的 GPT 架构：

```rust
pub struct GPTModel {
    token_embedding: Embedding,    // Token 嵌入层
    blocks: Vec<Block>,           // Transformer 块列表
    layer_norm: LayerNorm,        // 最终层归一化
    lm_head: Linear,              // 语言模型头
    // ... 其他组件
}
```

**核心功能**：
- **训练**：支持批次训练和梯度更新
- **推理**：支持两种生成模式（带/不带 KV 缓存）
- **模型保存/加载**：使用 SafeTensors 格式

#### 2.3 多头注意力机制 (`multi_head.rs`)

实现了 Multi-Head Attention 的核心逻辑：

```rust
pub struct MultiHeadAttention {
    heads: Vec<Head>,             // 多个注意力头
    proj: Linear,                 // 输出投影层
    rotary_emb: RotaryEmbedding, // 旋转位置编码
    // ... 其他字段
}
```

**关键特性**：
- **并行计算**：多头同时计算注意力
- **KV 缓存**：支持推理时的键值缓存优化
- **位置编码**：集成旋转位置编码（RoPE）

#### 2.4 注意力头 (`head.rs`)

单个注意力头的实现：
- **Q/K/V 计算**：查询、键、值矩阵的线性变换
- **注意力分数**：缩放点积注意力机制
- **因果掩码**：确保解码器的因果性质

#### 2.5 前馈网络 (`feed_forward.rs`)

实现了 Transformer 中的前馈神经网络：
- **两层线性变换**：维度扩展（4x）后收缩
- **激活函数**：使用 GELU 激活
- **Dropout**：防止过拟合

#### 2.6 旋转位置编码 (`rotary_emb.rs`)

实现了 RoPE（Rotary Position Embedding）：
- **相对位置**：捕捉 token 间的相对位置关系
- **旋转变换**：通过复数旋转编码位置信息
- **长度外推**：支持训练长度外的推理

### 3. 可执行程序

#### 3.1 分词器训练 (`tokenizer_train.rs`)

**功能**：
- **BPE 训练**：基于字节对编码算法
- **中文优化**：针对中文文本的特殊处理
- **词汇表生成**：创建模型专用的词汇表

**使用方法**：
```bash
cargo run --bin tokenizer_train
```

#### 3.2 模型训练 (`model_trainer.rs`)

**功能**：
- **端到端训练**：从数据加载到模型保存
- **实时监控**：训练过程的日志记录
- **优雅停止**：支持 Ctrl+C 中断并保存模型
- **设备自适应**：自动选择最佳计算设备

**使用方法**：
```bash
cargo run --bin train
```

#### 3.3 推理程序 (`inference.rs`)

**功能**：
- **文本生成**：基于输入提示生成新文本
- **性能对比**：比较有/无 KV 缓存的推理速度
- **参数调节**：支持温度采样等生成参数

**使用方法**：
```bash
cargo run --bin infer
```

## 🚀 快速开始

### 环境准备

1. **安装 Rust**：确保已安装 Rust 1.70+ 版本
2. **GPU 支持**（可选）：
   - CUDA：安装 CUDA 11.8+
   - Metal：macOS 系统自带

### 第一步：准备训练数据

将你的训练文本放入 `res/articles/` 目录：
- `tokenizer_train.txt` - 分词器训练数据
- `pretrain.txt` - 模型预训练数据

### 第二步：训练分词器

```bash
cargo run --bin tokenizer_train
```

这将生成 `mini_bpe.json` 文件，包含训练好的分词器。

### 第三步：训练模型

```bash
cargo run --bin train
```

训练过程中会生成：
- `config.json` - 模型配置文件
- `gpt_model.safetensors` - 训练好的模型权重

### 第四步：推理测试

```bash
cargo run --bin infer
```

## ⚙️ 配置说明

### 模型配置

在 `src/transformer/mod.rs` 中可以调整模型参数：

```rust
const CONTEXT_SIZE: usize = 64;    // 上下文长度
const N_VOCAB: usize = 22;         // 词汇表大小
const N_EMBED: usize = 32;         // 嵌入维度
const N_HEAD: usize = 8;           // 注意力头数
const N_LAYER: usize = 6;          // Transformer 层数
const DROPOUT: f32 = 0.1;          // Dropout 率
```

### 训练配置

在 `model_trainer.rs` 中可以调整训练参数：
- 批次大小
- 训练轮数
- 设备选择

## 🔧 技术特点

### 1. 高效的注意力机制
- **KV 缓存**：推理时避免重复计算
- **因果掩码**：确保解码器的自回归特性
- **缩放点积**：标准的注意力计算方式

### 2. 现代位置编码
- **RoPE**：旋转位置编码，优于传统绝对位置编码
- **相对位置**：更好地捕捉序列中的位置关系
- **长度外推**：支持超出训练长度的推理

### 3. 优化的数据流水线
- **批次处理**：高效的批次数据生成
- **内存管理**：合理的张量内存使用
- **设备优化**：自动选择最佳计算设备

### 4. 完整的工具链
- **分词器**：自定义 BPE 分词器训练
- **训练**：完整的模型训练流程
- **推理**：高效的文本生成功能

## 📊 性能特性

### KV 缓存优化
项目实现了 KV 缓存机制，在推理时可以显著提升性能：
- **缓存键值**：避免重复计算历史 token 的 K/V
- **内存效率**：合理管理缓存内存使用
- **速度提升**：长序列生成时性能提升明显

### 多设备支持
- **CPU**：支持纯 CPU 推理和训练
- **CUDA**：NVIDIA GPU 加速
- **Metal**：Apple Silicon 加速

## 🎯 使用场景

### 1. 教育学习
- **理解 Transformer**：清晰的代码结构便于学习
- **实验验证**：可以快速验证理论知识
- **参数调优**：方便调整参数观察效果

### 2. 研究开发
- **基础框架**：可作为更复杂模型的基础
- **算法验证**：快速验证新的算法想法
- **性能优化**：研究不同优化策略的效果

### 3. 实际应用
- **文本生成**：小规模文本生成任务
- **原型开发**：快速原型验证
- **边缘部署**：轻量级模型部署

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 改进方向
- **模型架构**：支持更多 Transformer 变体
- **训练策略**：更多训练技巧和优化方法
- **推理优化**：更高效的推理实现
- **文档完善**：更详细的使用文档

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

感谢 Candle 团队提供的优秀深度学习框架，以及 Rust 社区的各种优秀工具库。

---

*希望这个项目能帮助你更好地理解和使用 Transformer 架构！如果有任何问题，欢迎提交 Issue。* 