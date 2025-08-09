# Windows环境下的Unsloth框架LoRA微调通用模板

本项目提供了一个基于Unsloth框架在Windows环境下进行大语言模型LoRA微调的通用代码模板。当前示例展示了如何微调网络安全领域的专业模型，但您可以轻松替换基础模型和数据集来微调任何领域的模型。

## 🚀 快速开始

本项目支持任何兼容Unsloth的大语言模型微调，包括但不限于：
- **Llama系列**: Llama2, Llama3, Code Llama等
- **Qwen系列**: Qwen1.5, Qwen2, Qwen2.5等  
- **Mistral系列**: Mistral 7B, Mixtral等
- **Gemma系列**: Gemma 2B, 7B等
- **Yi系列**: Yi 6B, 34B等

当前示例使用网络安全数据集微调模型，您可以替换为任何领域的数据集。

## 📁 项目文件结构

- `train.py` - 训练脚本，负责模型微调（可配置任何模型和数据集）
- `inference.py` - 推理脚本，用于测试微调后的模型
- `unsloth-cli.py` - Unsloth命令行工具脚本（通用版本）
- `models/` - 基础模型目录（可放置任何支持的模型）
- `datasets/` - 训练数据集目录（支持多种数据格式）
- `outputs/` - 训练过程中的检查点和日志输出目录
- `gguf-py/` - GGUF格式转换工具库目录
- `env/` - Python虚拟环境目录

## 关键组件说明

### Unsloth

Unsloth是一个专为大语言模型(LLM)设计的高性能微调框架，它通过优化底层实现，能够使训练速度提升2-5倍，同时降低显存需求。本项目直接使用了Unsloth的核心功能进行模型微调。

### unsloth-cli.py

`unsloth-cli.py`是一个通用的命令行工具脚本，提供了Unsloth框架的主要功能的命令行接口。它允许用户通过命令行参数来配置和执行模型微调，而无需编写Python代码。主要特点包括：

- 提供多种命令行参数以配置模型加载、训练和保存
- 支持LoRA (Low-Rank Adaptation) 参数配置
- 内置数据集处理功能
- 支持模型保存为不同格式（包括GGUF）
- 支持将模型推送到Hugging Face Hub

使用示例：
```bash
# 使用命令行工具快速开始
python unsloth-cli.py --model_name "unsloth/llama-3-8b" --dataset "your_dataset" --save_model --save_gguf --quantization q4_k_m

# 或者使用自定义训练脚本
python train.py
```

本项目提供了两种使用方式：
1. **快速模式**: 使用`unsloth-cli.py`快速测试不同模型和数据集
2. **自定义模式**: 使用`train.py`进行深度自定义和特殊数据处理

### gguf-py

`gguf-py`是一个用于模型格式转换的工具库，支持将Hugging Face格式的模型转换为GGUF格式。GGUF(GPT-Generated Unified Format)是一种优化的模型格式，专为本地部署和推理而设计，被广泛用于llama.cpp等本地推理引擎中。

在本项目中，`gguf-py`主要用于：
- 将训练好的模型转换为GGUF格式
- 支持不同量化级别(q4_k_m, q8_0, f16等)的模型生成
- 实现模型压缩和优化以便在本地设备上运行

## 💻 环境要求

### 硬件要求
- **GPU**: NVIDIA显卡，建议显存8GB以上（支持4bit量化可降低要求）
- **内存**: 建议16GB以上
- **存储**: 至少20GB可用空间
- **系统**: Windows 10/11 64位

### 软件环境
- **Python**: 3.8-3.11（推荐3.10）
- **CUDA**: 11.8+ 或 12.x
- **Git**: 用于下载模型和数据集

### 核心依赖库
- `unsloth` - 高性能LLM微调框架
- `transformers` - Hugging Face transformers库
- `datasets` - 数据集处理库
- `trl` - 强化学习训练库
- `tensorboard` - 训练可视化工具
- `torch` - PyTorch深度学习框架

## 🔄 完整训练流程

### 方法一：使用预配置的训练脚本（推荐新手）

1. **快速开始**：
```bash
# 直接运行当前示例（网络安全模型）
python train.py
```

2. **自定义你的模型**：
   - 修改`train.py`中的模型路径
   - 替换数据集路径
   - 调整训练参数
   - 运行训练

### 方法二：使用命令行工具（推荐有经验用户）

```bash
# 使用任何支持的模型和数据集
python unsloth-cli.py \
    --model_name "unsloth/llama-3-8b" \
    --dataset "your_dataset_path" \
    --max_steps 100 \
    --save_model \
    --save_gguf \
    --quantization q4_k_m
```

### 自动化功能
- ✅ 自动保存微调后的模型
- ✅ 自动转换为GGUF格式（多种量化选项）
- ✅ 自动生成TensorBoard日志
- ✅ 自动处理显存优化

## 📊 数据集格式与自定义

### 支持的数据集格式

本项目支持多种数据集格式，当前示例使用网络安全数据集：

**格式1: 指令跟随格式（推荐）**
```json
[
    {
        "instruction": "你的任务指令",
        "input": "输入内容（可选）", 
        "output": "期望的输出"
    }
]
```

**格式2: 简单对话格式**
```json
[
    {
        "text": "用户: 问题内容\n助手: 回答内容"
    }
]
```

### 如何替换为你的数据集

1. **准备数据**：将你的数据整理为上述格式之一
2. **修改路径**：在`train.py`中修改数据集路径
3. **调整提示模板**：根据你的领域修改提示模板
4. **开始训练**：运行训练脚本

## 🎯 模型自定义指南

### 支持的基础模型

你可以使用任何兼容Unsloth的模型作为基础模型：

```python
# 在train.py中修改模型名称
model_name_options = [
    "unsloth/llama-3-8b",           # Llama 3 8B
    "unsloth/llama-3-70b",          # Llama 3 70B  
    "unsloth/mistral-7b",           # Mistral 7B
    "unsloth/qwen2-7b",             # Qwen2 7B
    "unsloth/gemma-7b",             # Gemma 7B
    "unsloth/yi-6b",                # Yi 6B
    # 或使用本地模型路径
    "./models/your-model-name"
]
```

### 训练流程说明

训练脚本会自动执行以下操作：
1. **模型加载**：加载指定的基础模型
2. **数据处理**：加载并格式化训练数据集
3. **LoRA配置**：应用LoRA（低秩适应）技术进行高效微调
4. **训练执行**：开始监督微调过程
5. **检查点保存**：定期保存训练检查点到`outputs`目录
6. **模型保存**：保存最终微调模型
7. **格式转换**：自动转换为GGUF格式（支持多种量化选项）

### 主要训练参数说明：

- `max_seq_length`: 2048，最大序列长度
- `r`: 16，LoRA秩
- `lora_alpha`: 16，LoRA缩放因子
- `per_device_train_batch_size`: 2，每个设备的批次大小
- `gradient_accumulation_steps`: 4，梯度累积步数
- `learning_rate`: 2e-4，学习率
- `max_steps`: 60，训练最大步数

### 训练监控

使用TensorBoard实时监控训练进度：
```bash
# 在新的命令行窗口中运行
python -m tensorboard.main --logdir=outputs

# 然后在浏览器中访问: http://localhost:6006
```

可监控的指标：
- 训练损失 (Training Loss)
- 学习率变化 (Learning Rate)
- 梯度范数 (Gradient Norm)  
- 训练步数和时间

## 🚀 模型推理使用

训练完成后，使用推理脚本测试你的模型：

```bash
# 使用微调后的模型（默认）
python inference.py --model_name ./your-fine-tuned-model

# 使用4位量化加载模型（节省显存）
python inference.py --model_name ./your-fine-tuned-model --load_in_4bit

# 对比原始基础模型效果
python inference.py --model_name ./models/your-base-model
```

### 推理脚本特点
- 支持命令行交互
- 支持流式输出
- 自动GPU加速
- 支持多种量化模式

## 📦 模型格式转换

### 自动转换（推荐）
训练脚本会自动生成GGUF格式模型，支持以下量化选项：

```python
# 在train.py中选择量化方式
quantization_options = [
    "f16",      # 半精度浮点（最高质量，文件较大）
    "q8_0",     # 8位量化（质量好，文件中等）  
    "q4_k_m",   # 4位量化（质量较好，文件最小）
    "q2_k",     # 2位量化（质量一般，文件极小）
]
```

### 手动转换
如需单独转换模型格式：

```bash
python -m gguf.gguf_converter \
    ./your-fine-tuned-model \
    ./your-model-gguf \
    -q q4_k_m
```

## ⚙️ 自定义配置指南

### 1. 更换基础模型

```python
# 在train.py中修改模型路径
model_options = {
    # Hugging Face模型
    "model_name": "unsloth/llama-3-8b",
    
    # 本地模型  
    "model_name": "./models/your-local-model",
    
    # 其他流行模型
    "model_name": "unsloth/mistral-7b-instruct-v0.3",
}
```

### 2. 替换训练数据集

```python
# 修改数据集路径和格式
dataset_options = [
    "datasets/your-custom-dataset",     # 本地数据集
    "username/dataset-name",            # Hugging Face数据集
    "path/to/json/files",              # JSON文件路径
]

# 在train.py中更新加载方式
dataset = load_dataset("your_dataset_path", split="train")
```

### 3. 自定义提示模板

```python
# 根据你的应用场景修改提示模板
custom_prompt_template = """你是一个专业的{领域}助手。

### 任务：
{instruction}

### 输入：  
{input}

### 回答：
{output}"""
```

### 4. 调整训练参数

根据你的硬件和需求调整参数：

```python
# 显存配置
load_in_4bit = True  # 低显存设备启用

# 训练超参数
training_config = {
    "learning_rate": 2e-4,              # 学习率
    "max_steps": 100,                   # 训练步数
    "per_device_train_batch_size": 2,   # 批次大小
    "gradient_accumulation_steps": 4,   # 梯度累积
}

# LoRA参数
lora_config = {
    "r": 16,                # LoRA秩（8-64）
    "lora_alpha": 16,       # 缩放因子
    "lora_dropout": 0.1,    # Dropout率
}
```

## ⚠️ 重要注意事项

### 硬件要求
- **必需**: NVIDIA GPU（支持CUDA）
- **显存**: 建议8GB以上，可通过4bit量化降低需求
- **内存**: 建议16GB以上系统内存

### 性能优化建议
1. **显存不足**: 启用`load_in_4bit=True`和减小批次大小
2. **加速训练**: 增加批次大小和梯度累积步数
3. **提高质量**: 增加训练步数和调整学习率
4. **模型收敛**: 监控TensorBoard确保损失下降

### 常见应用场景

**📚 教育领域**
- 替换为教育问答数据集
- 微调成专业课程助手

**💼 商业应用**  
- 客服对话数据集
- 行业专业知识库

**🔬 科研领域**
- 学术论文摘要生成
- 专业术语解释

**🎯 个人项目**
- 个人助手定制
- 创意写作辅助

## 📚 学习资源

### 官方文档
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)

### 社区资源
- Unsloth Discord社区
- Hugging Face论坛
- GitHub Issues交流 