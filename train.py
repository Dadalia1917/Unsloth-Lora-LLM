import os
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer  # 用于监督微调的训练器
from transformers import TrainingArguments  # 用于配置训练参数
from unsloth import is_bfloat16_supported  # 检查是否支持bfloat16精度训练
import wandb

# wandb.login(key="")  # 如果想启用wandb就取消注释，并将自己账号的key复制进去

if __name__ == '__main__':
    # 模型配置参数
    max_seq_length = 2048  # 最大序列长度
    dtype = None  # 数据类型，None表示自动选择
    load_in_4bit = False  # 使用4bit量化加载模型以节省显存

    # 加载预训练模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/Qwen3-1.7B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    train_prompt_style = """以下是描述任务的指令，以及提供更多上下文的输入。
                            请写出恰当完成该请求的回答。
                            在回答之前，请仔细思考问题，并创建一个逐步的思维链，以确保回答合乎逻辑且准确。

                            ### Instruction:
                            你是一位在网络安全、网络攻防、信息保护和安全架构设计方面具有专业知识的网络安全专家。
                            请回答以下网络安全相关问题。

                            ### Question:
                            {}

                            ### Response:
                            <think>
                            {}
                            </think>
                            {}"""

    EOS_TOKEN = tokenizer.eos_token # 添加结束符标记

    # 格式化提示函数,用于处理数据集中的示例
    def formatting_prompts_func(examples):
        # 从examples中提取问题和回答
        inputs = examples["instruction"]  # 网络安全问题列表
        outputs = examples["output"]  # 回答列表
        texts = []  # 存储格式化后的文本
        # 遍历每个示例,将问题和回答组合成指定格式
        for input, output in zip(inputs, outputs):
            # 为思维链部分提供空字符串，使用train_prompt_style模板格式化文本,并添加结束符
            # 提供三个参数：问题、思维链(空字符串)、回答
            text = train_prompt_style.format(input, "", output) + EOS_TOKEN
            texts.append(text)
        # 返回格式化后的文本字典
        return {
            "text": texts,
        }

    # 加载数据集并应用格式化
    dataset = load_dataset("datasets/NetworkSecurity", split="train",
                           trust_remote_code=True)
    dataset = dataset.map(formatting_prompts_func, batched=True, )

    model = FastLanguageModel.get_peft_model(
        # 原始模型
        model,
        # LoRA秩,用于控制低秩矩阵的维度,值越大表示可训练参数越多,模型性能可能更好但训练开销更大
        # 建议: 8-32之间
        r=16,
        # 需要应用LoRA的目标模块列表
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention相关层
            "gate_proj", "up_proj", "down_proj",  # FFN相关层
        ],
        # LoRA缩放因子,用于控制LoRA更新的幅度。值越大，LoRA的更新影响越大。
        lora_alpha=16,
        # LoRA层的dropout率,用于防止过拟合,这里设为0表示不使用dropout。
        # 如果数据集较小，建议设置0.1左右。
        lora_dropout=0.1,
        # 是否对bias参数进行微调,none表示不微调bias
        # none: 不微调偏置参数；
        # all: 微调所有参数；
        # lora_only: 只微调LoRA参数。
        bias="none",
        # 是否使用梯度检查点技术节省显存,使用unsloth优化版本
        # 会略微降低训练速度，但可以显著减少显存使用
        use_gradient_checkpointing="unsloth",
        # 随机数种子,用于结果复现
        random_state=3407,
        # 是否使用rank-stabilized LoRA,这里不使用
        # 会略微降低训练速度，但可以显著减少显存使用
        use_rslora=False,
        # LoFTQ配置,这里不使用该量化技术，用于进一步压缩模型大小
        loftq_config=None,
    )

    # 初始化SFT训练器
    trainer = SFTTrainer(
        model=model,  # 待训练的模型
        tokenizer=tokenizer,  # 分词器
        train_dataset=dataset,  # 训练数据集
        dataset_text_field="text",  # 数据集字段的名称
        max_seq_length=max_seq_length,  # 最大序列长度
        dataset_num_proc=2,  # 数据集处理的并行进程数，提高CPU利用率
        args=TrainingArguments(
            per_device_train_batch_size=2,  # 每个GPU的训练批次大小
            gradient_accumulation_steps=4,  # 梯度累积步数,用于模拟更大的batch size
            warmup_steps=5,  # 预热步数,逐步增加学习率
            learning_rate=2e-4,  # 学习率
            lr_scheduler_type="linear",  # 线性学习率调度器
            max_steps=60,  # 最大训练步数（一步 = 处理一个batch的数据）
            # 根据硬件支持选择训练精度
            fp16=not is_bfloat16_supported(),  # 如果不支持bf16则使用fp16
            bf16=is_bfloat16_supported(),  # 如果支持则使用bf16
            logging_steps=1,  # 每1步记录一次日志
            optim="adamw_8bit",  # 使用8位AdamW优化器节省显存，几乎不影响训练效果
            weight_decay=0.01,  # 权重衰减系数,用于正则化，防止过拟合
            seed=3407,  # 随机数种子
            output_dir="outputs",  # 保存模型检查点和训练日志
            save_strategy="steps",  # 按步保存中间权重
            save_steps=20,  # 每20步保存一次中间权重
            report_to="tensorboard",  # 将信息输出到tensorboard
        ),
    )

    # 开始训练，resume_from_checkpoint为True表示从最新的模型开始训练
    trainer_stats = trainer.train(resume_from_checkpoint = False)

    # 模型合并后生成的路径
    new_model_local = "Unsloth-Models"
    model.save_pretrained(new_model_local)  # 保存训练模型
    tokenizer.save_pretrained(new_model_local)  # 保存分词器

    # 保存合并后的16bit模型
    # model.save_pretrained_merged(new_model_local, tokenizer, save_method="merged_16bit", )

    # 保存gguf格式模型，根据需要自行取消注释
    model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="q4_k_m")
    # model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="q8_0")
    # model.save_pretrained_gguf(new_model_local, tokenizer, quantization_method="f16")