# Unsloth LoRA 纯文本微调通用教程

这是一个基于 Unsloth、Transformers、TRL 和 PEFT 的纯文本 SFT/LoRA 项目模板，适用于
Windows、Linux 和 WSL。代码使用模型原生 chat template、TRL `SFTConfig`、自动 LoRA
目标检测、仅助手回复 loss masking，并对 Qwen3、Qwen3.5/3.6、MoE 和 DeepSeek-V4 等
新架构进行前置兼容检查。

项目默认示例使用 `models/Qwen3-1.7B` 与 `datasets/NetworkSecurity`，但模型和数据集都可
替换为本地路径或 Hugging Face 仓库。本项目只训练文本部分，不训练视觉编码器。

## 功能概览

- 支持普通 LoRA、4bit/8bit 量化加载和 bf16/fp16 自动选择。
- 优先使用模型自带的 chat template，避免手写提示模板与模型格式不一致。
- 支持 Alpaca、OpenAI/ShareGPT、已有 `text` 列等常见数据格式。
- 默认只对 assistant 回复计算 loss，自动适配 thinking 与非 thinking 模板。
- 自动扫描注意力、MLP、融合 QKV、Gated DeltaNet 等可注入 LoRA 的模块。
- 支持融合 MoE 专家的 PEFT `target_parameters`，可限制专家层与专家 LoRA rank。
- 运行前检测 CUDA、bf16、bitsandbytes、flash-attn、FLA 等实际能力并自动降级。
- 支持断点续训、LoRA 保存、合并模型、GGUF 导出和多轮推理。

## 项目结构

| 文件或目录 | 作用 |
|---|---|
| `common.py` | 环境探测、模型识别、chat template、数据集和 LoRA/MoE 通用逻辑 |
| `train.py` | 集中配置式训练入口 |
| `unsloth-cli.py` | 命令行训练入口 |
| `inference.py` | LoRA/基础模型推理与多轮对话 |
| `datasets/` | 本地训练数据目录 |
| `models/` | 本地基础模型目录 |
| `outputs/` | checkpoint、日志和 TensorBoard 数据 |
| `Unsloth-Models/` | 默认 LoRA 适配器输出目录 |

## 环境要求与安装

建议使用独立的 Conda、venv 或 uv 环境，避免其他大模型项目对 torch、Transformers、TRL
等核心依赖提出相互冲突的版本要求。具体支持的 Python、PyTorch 和 CUDA 组合会随 Unsloth
更新，请优先参考 [Unsloth 官方安装文档](https://unsloth.ai/docs/get-started/install/pip-install)。

### Conda / venv

下面给出通用示例。已有可用环境时，不必为了本项目强制重装 torch 或 CUDA：

```shell
conda create -n unsloth-lora python=3.12 -y
conda activate unsloth-lora
python -m pip install --upgrade pip
pip install unsloth tensorboard
```

使用 uv 时，可让 uv 自动选择合适的 PyTorch 后端：

```shell
uv venv --python 3.13
uv pip install unsloth --torch-backend=auto
uv pip install tensorboard
```

如果需要自行安装 PyTorch，应先通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
生成与操作系统和 CUDA 匹配的命令，再安装 Unsloth。不要从旧教程复制固定 CUDA 版本的安装命令。

### 环境检查

```shell
python --version
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('available:', torch.cuda.is_available())"
python -c "import unsloth, transformers, trl, peft; print('Unsloth import OK')"
```

三个入口都会调用 `common.py` 输出运行时能力，包括 GPU、bf16、bitsandbytes、flash-attn
和线性注意力加速包。包已安装但原生库不可用时，也会按实际能力降级。

### 可选加速依赖

`flash-attn`、`causal-conv1d` 和 FLA 都不是本项目的硬依赖：

- `flash-attn` 用于普通全注意力层；不可用时可回退到 Unsloth、xFormers 或 PyTorch SDPA。
- Qwen3.5/3.6 的 Gated DeltaNet 线性注意力可使用 `causal-conv1d` 与 FLA 快速内核；
  缺失时 Transformers 可回退到较慢、显存开销更高的 PyTorch 实现。
- 这些包属于环境依赖，不需要把实现代码复制进本项目。安装成功后框架会自动发现。
- CUDA 扩展必须与操作系统、Python、PyTorch、CUDA 和 GPU 架构匹配；没有对应 wheel 时不要盲目源码编译。

在确认目标环境兼容后，可参考各自官方说明安装：

```shell
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention
```

参考：[Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)、
[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)。

## 支持的模型类型

| 模型类型 | 项目支持情况 | 使用提示 |
|---|---|---|
| Qwen3 稠密模型 | 原生模板、thinking 开关、LoRA/QLoRA | 根据显存选择精度、序列长度和 batch |
| Qwen3 MoE | 自动识别普通层与融合专家参数 | 默认不训练专家；需要时显式开启专家 LoRA |
| Qwen3.5/3.6 | Transformers v5、纯文本加载、混合线性注意力 | 默认 `text_only=True`，不加载视觉塔 |
| DeepSeek-V4 | 前置识别、版本检查、纯文本模板安全子集 | 需要含 `deepseek_v4` 的 Transformers/Unsloth 组合和相应计算资源 |
| Llama、Qwen2.5、Gemma、Mistral 等 | 原生模板和自动 LoRA 目标扫描 | 以模型配置和实际模块结构为准 |

DeepSeek-V4 原生架构在 Transformers 5.9 加入。若安装环境缺少该架构，代码会在加载权重前
给出明确错误。不要只升级单个 Transformers 包而忽略 Unsloth、TRL、PEFT 和 torch 的兼容关系。
DeepSeek-V4 的工具调用和完整官方协议应使用模型仓库提供的 encoder；本项目只提供纯文本 SFT
所需的基础模板子集。

## 快速开始

### 使用 `train.py`

直接执行：

```shell
python train.py
```

在 `train.py` 顶部修改集中配置：

```python
MODEL_NAME = "models/Qwen3-1.7B"       # 本地目录或 Hugging Face 仓库名
DATASET_SOURCE = "datasets/NetworkSecurity"
DATASET_SPLIT = "train"
MAX_SEQ_LENGTH = 2048

LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
TEXT_ONLY = True

ENABLE_THINKING = False
TRAIN_ON_RESPONSES_ONLY = True

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 60
NUM_TRAIN_EPOCHS = 1
```

`MAX_STEPS > 0` 时按步数训练，适合验证流程。按完整数据集训练时设置：

```python
MAX_STEPS = -1
NUM_TRAIN_EPOCHS = 1
```

### 使用 CLI

下面均为单行命令，可直接用于 PowerShell、CMD 或 Bash。

普通 LoRA：

```shell
python unsloth-cli.py --model_name models/Qwen3-1.7B --dataset datasets/NetworkSecurity --max_steps 60 --save_model
```

4bit QLoRA：

```shell
python unsloth-cli.py --model_name models/Qwen3-1.7B --dataset datasets/NetworkSecurity --load_in_4bit --per_device_train_batch_size 1 --save_model
```

Qwen3.5 纯文本 LoRA：

```shell
python unsloth-cli.py --model_name Qwen/Qwen3.5-0.8B --dataset datasets/NetworkSecurity --max_seq_length 2048 --max_steps -1 --num_train_epochs 1 --save_model
```

查看所有参数：

```shell
python unsloth-cli.py --help
```

## 数据集格式

数据源可以是本地目录、JSON/JSONL、CSV、Parquet、TXT 单文件或 Hugging Face 数据集名。
本地目录会递归寻找受支持的数据文件。

### Alpaca 格式

```json
[
  {
    "instruction": "什么是 SQL 注入？",
    "input": "",
    "output": "SQL 注入是……"
  }
]
```

也支持 `question/answer`、`prompt/completion`、`query/response` 等常见字段别名。

### OpenAI / ShareGPT 格式

```json
[
  {
    "messages": [
      {"role": "system", "content": "你是一名安全专家。"},
      {"role": "user", "content": "解释 CSRF。"},
      {"role": "assistant", "content": "CSRF 是……"}
    ]
  }
]
```

`conversations`、`conversation`、`chat`、`dialog` 也会被识别，ShareGPT 的
`human/gpt` 角色会自动转换为 `user/assistant`。

### 已格式化的 `text` 列

如果数据集已经包含 `text` 列，代码会认为它已经按目标模型的模板格式化，不会再次套用
chat template。此时必须自行保证模板与目标模型一致。

### Thinking 数据

独立思考字段支持 `reasoning`、`reasoning_content`、`thinking`、`cot`、`complex_cot` 等。
只有设置 `ENABLE_THINKING=True` 或添加 `--enable_thinking` 时，这些字段才会包装到
`<think>...</think>`。已经直接写在 assistant content 中的 `<think>` 内容不会被删除。

没有可靠推理过程的数据应保持非 thinking 模式。训练 thinking 模型时，应根据目标模型官方
建议混合推理与直答数据，并确保训练和推理使用相同的 chat template。

## LoRA 与 MoE 配置

### LoRA 参数

| 参数 | 说明 |
|---|---|
| `LORA_R` / `--r` | LoRA rank；越大可训练参数越多 |
| `LORA_ALPHA` / `--lora_alpha` | LoRA 缩放因子 |
| `LORA_DROPOUT` / `--lora_dropout` | 默认为 0，可使用 Unsloth 优化内核 |
| `LORA_TARGET_MODULES` / `--target_modules` | 留空时按模型实际结构自动检测 |
| `USE_RSLORA` / `--use_rslora` | 启用 rank-stabilized LoRA |

自动扫描默认排除 embedding、输出头、norm、视觉层和 MoE router，并识别常见注意力、MLP、
融合 QKV、Gated DeltaNet 投影层。只有模型结构特殊且自动扫描结果不正确时，才需要手动指定
`target_modules`。

### MoE 专家

MoE 默认只训练普通注意力和共享线性层，不训练全部专家：

```python
TRAIN_MOE_EXPERTS = False
MOE_EXPERT_LAYERS = None
MOE_EXPERT_RANK = None
```

CLI 示例：

```shell
python unsloth-cli.py --model_name your-moe-model --dataset your-dataset --train_moe_experts --moe_expert_layers 20 21 22 --moe_expert_rank 2 --save_model
```

新版融合专家通常是 3D `nn.Parameter`，项目通过 PEFT `target_parameters` 注入 LoRA，
并默认按专家数量缩小 rank，避免可训练参数量随专家数快速增长。

## 训练、恢复与监控

训练会在 `OUTPUT_DIR` / `--output_dir` 中保存 checkpoint。TensorBoard：

```shell
python -m tensorboard.main --logdir=outputs
```

从最新 checkpoint 恢复：

```shell
python unsloth-cli.py --resume_from_checkpoint
```

从指定 checkpoint 恢复：

```shell
python unsloth-cli.py --resume_from_checkpoint outputs/checkpoint-40
```

恢复时应保持基础模型、LoRA 目标模块和主要训练配置一致。

## 推理

加载默认 `Unsloth-Models` LoRA 并进入多轮对话：

```shell
python inference.py
```

常见用法：

```shell
python inference.py --question "解释一下 XSS 与 CSRF 的区别"
python inference.py --thinking
python inference.py --model_name models/Qwen3-1.7B
python inference.py --model_name Unsloth-Models --base_model models/Qwen3-1.7B
python inference.py --load_in_4bit --max_seq_length 4096 --max_new_tokens 1024
```

`--model_name` 可以是 LoRA 目录、合并模型目录、本地基础模型或 Hugging Face 仓库名。
通常直接传 LoRA 目录即可；只有 `adapter_config.json` 记录的基础模型路径失效时，才需要
`--base_model`。

多轮历史超过输入预算时，脚本从左侧截断，优先保留最新对话。交互命令：

- `/reset`：清空历史；
- `/exit` 或 `/quit`：退出。

旧脚本如果使用手写 `### Instruction/Response` 模板训练，而新版推理使用模型原生模板，效果会
明显受影响。这类适配器建议使用新版数据管线重新训练。

## 保存与导出

`train.py` 默认只保存 LoRA。训练末尾保留了与原版相同的注释式导出配置：

```python
SAVE_LORA_DIR = "Unsloth-Models"
GGUF_DIR = "Unsloth-Models-GGUF"

# model.save_pretrained_merged("Unsloth-Models-merged", tokenizer, save_method="merged_16bit")

gguf_quants = [
    # "q4_k_m",
    # "q8_0",
    # "f16",
]
```

默认三项都被注释，不会导出 GGUF。需要 Q4_K_M 时只删除 `"q4_k_m"` 前面的 `#`；需要多个
格式就同时取消多项注释。代码会把列表一次传给 Unsloth，不会为每种格式重复合并 LoRA。

CLI 示例：

```shell
python unsloth-cli.py --save_model --save_merged
python unsloth-cli.py --save_gguf --gguf_path Unsloth-Models-GGUF --quantization q4_k_m q8_0
```

LoRA、合并模型和 GGUF 使用不同目录。Unsloth 可能给最终目录添加 `_gguf` 后缀，训练脚本会
打印实际生成的每个文件，不要只根据传入的工作目录判断是否成功。

### 为什么旧代码只能生成 BF16，不能生成 Q4_K_M

GGUF 转换和量化是两个步骤：

```text
Hugging Face/合并后的 LoRA
        ↓ convert_hf_to_gguf.py
BF16 或 F16 GGUF
        ↓ llama-quantize
Q4_K_M GGUF
```

`convert_hf_to_gguf.py` 的 `--outtype` 只支持直接写出的格式，例如 `f32`、`f16`、`bf16`
和 `q8_0`；`q4_k_m` 不是它的合法 `--outtype`。K-quant 必须再调用 llama.cpp 编译得到的
`llama-quantize`（Windows 为 `llama-quantize.exe`）。所以 BF16 成功只能说明第一步正常，
不能说明第二步的量化器存在且可运行。

一年前常见的失败原因是：

- Windows 下没有编译出 `llama-quantize.exe`，或者缺少同目录的 `ggml`/`llama` DLL；
- Unsloth 只检查旧的量化器路径，没有找到 `build/bin/Release` 下的 Windows 程序；
- 项目里的 llama.cpp 版本早于目标模型支持，转换器和量化器版本不一致；
- 直接把 `q4_k_m` 传给 `convert_hf_to_gguf.py --outtype`；
- 中间 GGUF、最终 GGUF同时占用磁盘，或量化阶段系统内存不足。

llama.cpp 官方也把流程分成“先转换高精度 GGUF，再用 `llama-quantize` 量化”两步，参见
[llama.cpp quantize 文档](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize)。

### 新版代码如何处理

当前代码直接调用 Unsloth 的 `save_pretrained_gguf`。对 `q4_k_m`，新版 Unsloth 会：

1. 合并 LoRA 与基础模型；
2. 生成 BF16/F16 中间 GGUF；
3. 查找可运行的 `llama-quantize`，Windows 也会检查 `.exe` 和 `build/bin/Release`；
4. 量化为 Q4_K_M；
5. 成功后返回实际文件路径，并在只要求 Q4_K_M 时清理中间 GGUF。

多个量化格式应一次传入列表。这样只合并和转换一次，再从同一个高精度 GGUF生成各量化版本：

```python
gguf_quants = ["bf16", "q8_0", "q4_k_m"]
```

如果只需要部署文件，只取消 `"q4_k_m"` 的注释即可。不要先把 Q8_0 再量化成
Q4_K_M；应始终从 BF16/F16 中间文件量化，避免重复量化造成额外精度损失。Unsloth 官方接口
也直接支持 `q4_k_m`：[Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)。

新版 Unsloth 默认在用户目录的 `.unsloth/llama.cpp` 中准备工具链。为兼容旧项目，如果默认
目录尚不可用，而启动目录中存在可以运行的 `./llama.cpp`，Unsloth 也可能直接使用这份副本。
因此，即使没有设置环境变量，项目根目录中的旧版本也可能被选中。

新模型应更新、移走或重命名项目中的旧 `llama.cpp`，让 Unsloth 准备当前工具链。若要使用自己
维护且已经确认兼容的版本，可在启动 Python 前指定它的绝对路径：

```powershell
$env:UNSLOTH_LLAMA_CPP_PATH = "C:\tools\llama.cpp"
```

只有确认转换器支持目标模型、`llama-quantize.exe` 可以运行且所需 DLL 齐全时才这样做。转换
脚本、`gguf-py`、量化程序和 DLL 应来自同一版本，不要只替换其中一个文件。

### 手动量化

自动导出失败时，可以先保存合并后的 16bit 模型，再使用一套当前且兼容的 llama.cpp 手动执行
两步。Windows PowerShell 示例：

```powershell
$llamaCpp = "C:\tools\llama.cpp"

python "$llamaCpp\convert_hf_to_gguf.py" .\Unsloth-Models-merged `
  --outfile .\Unsloth-Models.BF16.gguf --outtype bf16

& "$llamaCpp\llama-quantize.exe" `
  .\Unsloth-Models.BF16.gguf `
  .\Unsloth-Models.Q4_K_M.gguf `
  Q4_K_M
```

Linux/WSL 将最后一条命令中的程序路径改为实际构建位置，通常是
`./llama.cpp/build/bin/llama-quantize`。若提示不支持模型架构，应先更新 llama.cpp，而不是
修改 GGUF 文件或强行使用 `--allow-requantize`。

## 自动兼容与降级

`common.py` 会在不修改 site-packages 的前提下处理或提示常见兼容问题：

- Windows 非 UTF-8 locale：入口脚本自动以 UTF-8 模式重启，避免 GBK 解码失败。
- Python 3.14 + `datasets<4.4`：运行时回移官方 Pickler 签名修复。
- bitsandbytes 原生库不可用：量化加载回退到 16bit，8bit 优化器回退到 torch 优化器。
- 显式请求 flash-attn 但无法导入：交给 Unsloth 自动选择其他注意力实现。
- 新模型架构不在当前 Transformers 中：在下载权重前报出明确的版本要求。
- 模型没有 chat template：只在缺失时使用 fallback，不覆盖模型原生模板。

项目能够自动降级不代表所有依赖组合都经过上游测试。自定义 torch/CUDA wheel 或混合安装多个
大模型框架时，应先检查：

```shell
pip check
```

## 常见问题

| 问题 | 处理建议 |
|---|---|
| CUDA OOM | 降低 batch、序列长度或 LoRA rank，增加梯度累积；适用模型可尝试 4bit |
| 未检测到 CUDA | 检查驱动、PyTorch CUDA 构建及 `torch.cuda.is_available()` |
| bitsandbytes DLL/so 加载失败 | 安装与系统、CUDA 和 torch 匹配的版本，或关闭 4/8bit |
| Qwen3.5 线性注意力较慢 | 检查 FLA/causal-conv1d 是否兼容；不兼容时使用回退并降低显存配置 |
| 数据集字段无法识别 | 转为 `text`、对话格式或 `instruction/input/output` |
| assistant 标签全部被屏蔽 | 检查 chat template、thinking 模式和序列截断 |
| torch.compile/Triton 报错 | CLI 添加 `--disable_compile`；必要时检查 Triton 与 torch 版本 |
| 模型要求远程代码 | 仅在确认仓库可信后使用 `--trust_remote_code` |
| 训练效果不佳 | 检查数据质量、模板一致性、学习率、训练步数及 response-only masking |
| GGUF 导出失败 | 确认 llama.cpp/Unsloth 已支持该模型，并检查本地编译工具链 |

## 参考资料

- [Unsloth 文档](https://unsloth.ai/docs)
- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Qwen3.6 官方仓库](https://github.com/QwenLM/Qwen3.6)
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [DeepSeek-V4 模型集合](https://huggingface.co/collections/deepseek-ai/deepseek-v4)

## 许可

请遵守基础模型、数据集、Unsloth、Transformers、TRL、PEFT 及其他依赖各自的许可证和使用条款。
