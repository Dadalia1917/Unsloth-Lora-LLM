from __future__ import annotations

# 必须先设置 UTF-8 和 Unsloth 环境，再导入训练库。
import common

common.apply_runtime_env()

from unsloth import FastLanguageModel  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


# 模型和数据
MODEL_NAME = "models/Qwen3-1.7B"
DATASET_SOURCE = "datasets/NetworkSecurity"
DATASET_SPLIT = "train"
DATASET_FILES = None
MAX_SAMPLES = None

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
ATTN_IMPLEMENTATION = "auto"
DEVICE_MAP = "sequential"
TRUST_REMOTE_CODE = False
TEXT_ONLY = True

SYSTEM_PROMPT = (
    "你是一位在网络安全、网络攻防、信息保护和安全架构设计方面具有专业知识的网络安全专家。"
    "请基于事实、分步骤地回答用户的网络安全问题。"
)
ENABLE_THINKING = False
TRAIN_ON_RESPONSES_ONLY = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_BIAS = "none"
LORA_TARGET_MODULES = None  # None 表示按模型结构自动查找
USE_RSLORA = False
USE_GRADIENT_CHECKPOINTING = "unsloth"

# MoE 默认只训练注意力和共享线性层
TRAIN_MOE_EXPERTS = False
MOE_EXPERT_LAYERS = None
MOE_EXPERT_RANK = None

# 训练参数
OUTPUT_DIR = "outputs"
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "linear"
WEIGHT_DECAY = 0.01
MAX_STEPS = 60  # 设为 -1 时按 NUM_TRAIN_EPOCHS 训练
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 20
OPTIMIZER = "adamw_8bit"
REPORT_TO = "tensorboard"
DATASET_NUM_PROC = 1
PACKING = False
RESUME_FROM_CHECKPOINT = False
SEED = 3407

# 保存
SAVE_LORA_DIR = "Unsloth-Models"
GGUF_DIR = "Unsloth-Models-GGUF"


def check_config() -> None:
    if min(MAX_SEQ_LENGTH, LORA_R, LORA_ALPHA, DATASET_NUM_PROC) <= 0:
        raise ValueError("序列长度、LoRA 参数和数据处理进程数必须大于 0")
    if not 0 <= LORA_DROPOUT < 1:
        raise ValueError("LORA_DROPOUT 必须在 [0, 1) 范围内")
    if MOE_EXPERT_RANK is not None and MOE_EXPERT_RANK <= 0:
        raise ValueError("MOE_EXPERT_RANK 必须大于 0")


def get_moe_options(model) -> dict:
    if not common.is_moe_model(model):
        return {}
    if not TRAIN_MOE_EXPERTS:
        print("检测到 MoE 模型，跳过专家参数")
        return {}

    target_parameters = common.auto_moe_target_parameters(model, MOE_EXPERT_LAYERS)
    if not target_parameters:
        print("没有找到融合专家参数，只训练普通线性层")
        return {}

    rank_pattern = common.moe_expert_rank_pattern(model, target_parameters, LORA_R)
    if MOE_EXPERT_RANK is not None:
        rank_pattern = {name: MOE_EXPERT_RANK for name in target_parameters}

    options = {"target_parameters": target_parameters}
    if rank_pattern:
        options["rank_pattern"] = rank_pattern
    print(f"训练 {len(target_parameters)} 个融合专家参数")
    return options


def main() -> None:
    check_config()

    capabilities = common.detect_capabilities()
    common.print_capabilities(capabilities)
    if not capabilities.gpu_name:
        raise RuntimeError("没有检测到可用的 NVIDIA GPU")

    profile = common.inspect_model_profile(MODEL_NAME, trust_remote_code=TRUST_REMOTE_CODE)
    common.check_model_requirements(profile)
    load_in_4bit, load_in_8bit = common.pick_quantization(
        capabilities, LOAD_IN_4BIT, LOAD_IN_8BIT
    )
    common.warn_model_runtime(profile, capabilities, load_in_4bit, load_in_8bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        load_in_16bit=not (load_in_4bit or load_in_8bit),
        full_finetuning=False,
        trust_remote_code=TRUST_REMOTE_CODE,
        device_map=DEVICE_MAP,
        text_only=TEXT_ONLY,
        **common.resolve_attn_implementation(capabilities, ATTN_IMPLEMENTATION),
    )
    tokenizer = common.ensure_chat_template(tokenizer, model_profile=profile)

    dataset = common.load_any_dataset(
        DATASET_SOURCE,
        split=DATASET_SPLIT,
        data_files=DATASET_FILES,
    )
    if MAX_SAMPLES is not None:
        if MAX_SAMPLES <= 0:
            raise ValueError("MAX_SAMPLES 必须大于 0")
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("数据集为空")

    print(f"数据集字段：{dataset.column_names}，样本数：{len(dataset)}")
    dataset = common.build_text_dataset(
        dataset,
        tokenizer,
        system_prompt=SYSTEM_PROMPT,
        enable_thinking=ENABLE_THINKING,
        num_proc=DATASET_NUM_PROC,
    )
    print("第一条训练样本：")
    print(dataset[0]["text"][:1200])

    target_modules = LORA_TARGET_MODULES or common.auto_lora_target_modules(
        model, text_only=TEXT_ONLY
    )
    if not target_modules:
        raise RuntimeError("没有找到 LoRA 目标模块，请手动设置 LORA_TARGET_MODULES")
    print(f"LoRA 目标模块：{target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=target_modules,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=USE_RSLORA,
        loftq_config=None,
        **get_moe_options(model),
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=PACKING,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        weight_decay=WEIGHT_DECAY,
        fp16=not capabilities.supports_bf16,
        bf16=capabilities.supports_bf16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        optim=common.pick_optimizer(capabilities, OPTIMIZER),
        seed=SEED,
        report_to=REPORT_TO,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    if TRAIN_ON_RESPONSES_ONLY:
        markers = common.detect_response_markers(tokenizer, ENABLE_THINKING)
        if markers:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part=markers[0],
                response_part=markers[1],
            )
            print(f"只计算助手回复的损失：{markers[1]!r}")
        else:
            print("没有识别出角色标记，将计算完整序列的损失")

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    model.save_pretrained(SAVE_LORA_DIR)
    tokenizer.save_pretrained(SAVE_LORA_DIR)
    print(f"LoRA 已保存到 {SAVE_LORA_DIR}")

    # 保存合并后的 16bit 模型时取消下一行注释
    # model.save_pretrained_merged("Unsloth-Models-merged", tokenizer, save_method="merged_16bit")

    # 取消对应注释即可导出一种或多种 GGUF
    gguf_quants = [
        # "q4_k_m",
        # "q8_0",
        # "f16",
    ]
    if gguf_quants:
        result = model.save_pretrained_gguf(
            GGUF_DIR,
            tokenizer,
            quantization_method=gguf_quants,
        )
        if isinstance(result, dict):
            for path in result.get("gguf_files", []):
                print(f"GGUF 已保存到 {path}")
        else:
            print(f"GGUF 导出完成，请检查 {GGUF_DIR}_gguf")


if __name__ == "__main__":
    main()
