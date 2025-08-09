# Windows环境下的Unsloth框架LoRA微调大模型完整教程

## 📋 目录
1. [项目简介](#项目简介)
2. [环境要求](#环境要求)  
3. [快速开始](#快速开始)
4. [详细配置说明](#详细配置说明)
5. [训练过程](#训练过程)
6. [模型推理](#模型推理)
7. [常见问题解决](#常见问题解决)
8. [进阶自定义](#进阶自定义)

---

## 🎯 项目简介

本教程基于Windows环境，使用Unsloth框架对大语言模型进行LoRA (Low-Rank Adaptation) 微调。Unsloth是一个专为大语言模型设计的高性能微调框架，相比传统方法可以提升2-5倍训练速度，同时显著降低显存需求。

### 🔥 主要特点
- **高效训练**：相比传统微调方法快2-5倍
- **显存优化**：通过多种优化技术大幅降低显存需求
- **Windows友好**：专门优化的Windows环境支持
- **一键运行**：提供完整的训练和推理脚本
- **格式多样**：支持输出GGUF等多种模型格式

---

## 💻 环境要求

### 硬件要求
- **GPU**: NVIDIA显卡，建议显存8GB以上
- **内存**: 建议16GB以上
- **存储**: 至少20GB可用空间

### 软件要求
- **操作系统**: Windows 10/11 64位
- **Python**: 3.8-3.11 (推荐3.10)
- **CUDA**: 11.8+ 或 12.x
- **Git**: 用于克隆代码库

### 核心依赖包
```
torch>=2.1.0
unsloth
transformers
datasets
trl
tensorboard
peft
```

---

## 🚀 快速开始

### 第一步：环境检查
打开PowerShell或命令提示符，检查Python和CUDA环境：

```powershell
# 检查Python版本
python --version

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA版本: {torch.version.cuda}')"
python -c "import torch; print(f'GPU设备数: {torch.cuda.device_count()}')"
```

### 第二步：安装依赖
```powershell
# 安装PyTorch (根据你的CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装Unsloth
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
# 或者对于CUDA 12.1+
# pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# 安装其他依赖
pip install transformers datasets trl tensorboard accelerate peft
```

### 第三步：下载项目文件
如果你已经有了项目文件，可以直接进入项目目录。项目应包含以下核心文件：
- `train.py` - 训练脚本
- `inference.py` - 推理脚本  
- `unsloth-cli.py` - 命令行工具
- `models/` - 基础模型目录
- `datasets/` - 训练数据目录

---

## ⚙️ 详细配置说明

### 训练配置参数详解

#### 模型配置
```python
max_seq_length = 2048      # 最大序列长度，影响内存使用
dtype = None               # 数据类型，None为自动选择
load_in_4bit = False       # 是否启用4位量化加载
```

#### LoRA参数配置
```python
r = 16                     # LoRA秩，控制可训练参数数量 (建议: 8-32)
lora_alpha = 16           # LoRA缩放因子，控制更新幅度
lora_dropout = 0.1        # Dropout率，防止过拟合
bias = "none"             # 偏置微调策略
```

#### 训练参数配置
```python
per_device_train_batch_size = 2    # 每GPU批次大小
gradient_accumulation_steps = 4     # 梯度累积步数
warmup_steps = 5                   # 预热步数
learning_rate = 2e-4               # 学习率
max_steps = 60                     # 最大训练步数
```

### 显存优化配置

#### 低显存配置 (8GB以下)
```python
load_in_4bit = True               # 启用4位量化
per_device_train_batch_size = 1   # 减小批次大小
gradient_accumulation_steps = 8   # 增加梯度累积
r = 8                            # 降低LoRA秩
```

#### 标准配置 (8-16GB显存)
```python
load_in_4bit = False              # 正常精度
per_device_train_batch_size = 2   # 标准批次大小
gradient_accumulation_steps = 4   # 标准梯度累积
r = 16                           # 标准LoRA秩
```

#### 高性能配置 (16GB+显存)
```python
load_in_4bit = False              # 正常精度
per_device_train_batch_size = 4   # 更大批次
gradient_accumulation_steps = 2   # 减少梯度累积
r = 32                           # 更高LoRA秩
```

---

## 🔄 训练过程

### 启动训练

**方法一：直接运行Python脚本**
```powershell
# 进入项目目录
cd D:\unsloth-Lora

# 启动训练
python train.py
```

**方法二：使用命令行工具**
```powershell
python unsloth-cli.py --model_name "models/Qwen3-1.7B" \
    --dataset "datasets/NetworkSecurity" \
    --save_model --save_gguf \
    --quantization q4_k_m \
    --max_steps 60
```

### 训练监控

**启动TensorBoard监控**
```powershell
# 新开一个PowerShell窗口
cd D:\unsloth-Lora
python -m tensorboard.main --logdir=outputs

# 然后在浏览器中访问: http://localhost:6006
```

**Windows批处理文件 (可选)**
```batch
@echo off
title "Unsloth微调训练"
python train.py
pause
```

### 训练过程说明

1. **模型加载阶段**
   - 加载基础模型 (Qwen3-1.7B)
   - 应用LoRA配置
   - 初始化训练器

2. **数据处理阶段**
   - 加载网络安全数据集
   - 格式化训练提示
   - 准备训练数据

3. **训练阶段**
   - 执行监督微调 (SFT)
   - 每20步保存检查点
   - 记录训练指标

4. **模型保存阶段**
   - 保存微调后的模型
   - 自动转换为GGUF格式
   - 生成可部署的模型文件

---

## 🤖 模型推理

### 基础推理使用

```powershell
# 使用微调后的模型
python inference.py --model_name Unsloth-Qwen3

# 使用4位量化加载 (节省显存)
python inference.py --model_name Unsloth-Qwen3 --load_in_4bit

# 使用原始基础模型对比
python inference.py --model_name models/Qwen3-1.7B
```

### 推理脚本使用示例

运行推理脚本后，会提示输入问题：
```
请输入您的问题：什么是SQL注入攻击？
```

系统会基于网络安全数据集的微调结果给出专业回答。

### 集成到其他应用

```python
import sys
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import TextStreamer

# 加载模型
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/Qwen3-1.7B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = PeftModel.from_pretrained(base_model, "Unsloth-Qwen3")
FastLanguageModel.for_inference(model)

# 推理函数
def chat_with_model(question):
    inputs = tokenizer([question], return_tensors="pt").to("cuda")
    text_streamer = TextStreamer(tokenizer)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=1200,
            use_cache=True,
            temperature=0.7,
            do_sample=True,
        )
    return outputs
```

---

## ❗ 常见问题解决

### 1. CUDA相关问题

**问题**: `CUDA out of memory`
**解决方案**:
```python
# 在train.py中修改以下参数
load_in_4bit = True                    # 启用4位量化
per_device_train_batch_size = 1        # 减小批次大小  
gradient_accumulation_steps = 8        # 增加梯度累积
```

**问题**: `No CUDA devices available`
**解决方案**:
1. 检查NVIDIA驱动是否正确安装
2. 验证CUDA工具包版本匹配
3. 重新安装对应CUDA版本的PyTorch

### 2. 安装相关问题

**问题**: `unsloth` 安装失败
**解决方案**:
```powershell
# 清理pip缓存
pip cache purge

# 使用清华源镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple unsloth

# 或者从源码安装
pip install git+https://github.com/unslothai/unsloth.git
```

**问题**: `Microsoft Visual C++ 14.0 is required`
**解决方案**:
1. 安装 Microsoft C++ Build Tools
2. 或安装 Visual Studio Community 2019/2022

### 3. 训练相关问题

**问题**: 训练速度过慢
**解决方案**:
- 检查是否启用了GPU训练
- 调整 `dataset_num_proc` 参数
- 考虑使用更小的模型进行测试

**问题**: 模型收敛效果不佳
**解决方案**:
- 增加训练步数 (`max_steps`)
- 调整学习率 (`learning_rate`)
- 检查数据集质量和格式

### 4. Windows特定问题

**问题**: 路径包含中文字符导致错误
**解决方案**:
- 使用英文路径
- 或在Python脚本开头添加：
```python
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
```

**问题**: PowerShell执行策略限制
**解决方案**:
```powershell
# 以管理员身份运行PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🔧 进阶自定义

### 自定义数据集

1. **准备数据格式**
```json
[
    {
        "instruction": "你的指令",
        "input": "输入内容",
        "output": "期望输出"
    }
]
```

2. **修改数据加载代码**
```python
# 在train.py中修改
dataset = load_dataset("path/to/your/dataset", split="train")
```

### 自定义提示模板

```python
# 修改train.py中的提示模板
custom_prompt_style = """你是一个专业的AI助手。

### 用户问题:
{}

### 回答:
{}"""
```

### 模型部署优化

**生成不同量化版本**
```python
# 在train.py末尾添加
model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="f16")    # 最高质量
model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="q8_0")   # 平衡质量/大小
model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="q4_k_m") # 最小大小
```

### 高级训练配置

**启用混合精度训练**
```python
from transformers import TrainingArguments

args = TrainingArguments(
    bf16=True,  # 启用BF16混合精度
    dataloader_pin_memory=False,  # Windows环境建议关闭
    remove_unused_columns=False,
)
```

**集成WandB监控**
```python
import wandb

# 在train.py开头添加
wandb.login(key="your_wandb_key")

# 在TrainingArguments中添加
report_to="wandb"
```

---

## 📚 参考资源

### 官方文档
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth 文档](https://docs.unsloth.ai/)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

### 相关教程
- [哔哩哔哩视频教程](https://www.bilibili.com/video/BV14YNReqEHu/)
- [Hugging Face LoRA 指南](https://huggingface.co/docs/peft/conceptual_guides/lora)

### 社区支持
- Unsloth Discord 社区
- GitHub Issues 反馈

---

## ⚖️ 许可声明

本教程仅供学习和研究使用。使用过程中请遵守相关开源协议和法律法规。

**最后更新**: 2024年1月
**教程版本**: v1.0

---

*🎉 恭喜！你已经掌握了在Windows环境下使用Unsloth框架进行LoRA微调的完整流程。开始你的AI模型微调之旅吧！*
