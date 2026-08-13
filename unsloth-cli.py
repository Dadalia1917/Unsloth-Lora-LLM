#!/usr/bin/env python3
"""Unsloth LoRA 命令行训练工具。"""

from __future__ import annotations

import argparse

# 必须先设置 UTF-8 和 Unsloth 环境。
import common


def run(args: argparse.Namespace) -> None:
    if args.max_seq_length <= 0:
        raise ValueError("--max_seq_length 必须大于 0")
    if args.r <= 0 or args.lora_alpha <= 0:
        raise ValueError("--r 和 --lora_alpha 必须大于 0")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora_dropout 必须在 [0, 1) 范围内")
    if args.moe_expert_rank is not None and args.moe_expert_rank <= 0:
        raise ValueError("--moe_expert_rank 必须大于 0")
    if args.dataset_num_proc <= 0:
        raise ValueError("--dataset_num_proc 必须大于 0")

    common.apply_runtime_env(compile_disable=args.disable_compile)

    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    caps = common.detect_capabilities()
    common.print_capabilities(caps)
    if not caps.gpu_name:
        raise RuntimeError("未检测到 CUDA GPU；Unsloth LoRA 训练需要可用的 NVIDIA GPU")

    profile = common.inspect_model_profile(args.model_name, trust_remote_code=args.trust_remote_code)
    common.check_model_requirements(profile)

    load_in_4bit, load_in_8bit = common.pick_quantization(caps, args.load_in_4bit, args.load_in_8bit)
    common.warn_model_runtime(profile, caps, load_in_4bit, load_in_8bit)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        load_in_16bit=not (load_in_4bit or load_in_8bit),
        full_finetuning=False,
        trust_remote_code=args.trust_remote_code,
        device_map=args.device_map,
        text_only=True,
        **common.resolve_attn_implementation(caps, args.attn_implementation),
    )
    tokenizer = common.ensure_chat_template(
        tokenizer,
        fallback_template=args.fallback_chat_template,
        model_profile=profile,
    )

    dataset = common.load_any_dataset(args.dataset, split=args.dataset_split, data_files=args.data_files)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max_samples 必须是正整数")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("数据集为空，无法训练")
    print(f"[数据] 原始字段：{dataset.column_names}，样本数：{len(dataset)}")

    dataset = common.build_text_dataset(
        dataset,
        tokenizer,
        system_prompt=args.system_prompt,
        enable_thinking=args.enable_thinking,
        num_proc=args.dataset_num_proc,
    )
    print("[数据] 模板渲染后的第一条样本：")
    print(dataset[0]["text"][:1200])

    target_modules = args.target_modules or common.auto_lora_target_modules(model, text_only=True)
    if not target_modules:
        raise RuntimeError("没有检测到可注入 LoRA 的线性层，请通过 --target_modules 显式指定")
    print(f"[LoRA] 目标模块：{target_modules}")

    peft_kwargs = {}
    if common.is_moe_model(model):
        if args.train_moe_experts:
            target_parameters = common.auto_moe_target_parameters(model, args.moe_expert_layers)
            if target_parameters:
                peft_kwargs["target_parameters"] = target_parameters
                rank_pattern = common.moe_expert_rank_pattern(model, target_parameters, args.r)
                if args.moe_expert_rank is not None:
                    rank_pattern = {name: args.moe_expert_rank for name in target_parameters}
                if rank_pattern:
                    peft_kwargs["rank_pattern"] = rank_pattern
                    print(f"[LoRA] 融合专家秩：{next(iter(rank_pattern.values()))}")
                print(f"[LoRA] 检测到 MoE 模型，额外训练 {len(target_parameters)} 个专家权重")
            else:
                print("[提示] 检测到 MoE，但没有发现 3D 融合专家参数；仅训练普通线性层")
        else:
            print("[LoRA] 检测到 MoE 模型，已跳过专家层（如需训练请加 --train_moe_experts）")

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        max_seq_length=args.max_seq_length,
        use_rslora=args.use_rslora,
        loftq_config=None,
        **peft_kwargs,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=args.packing,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        fp16=not caps.supports_bf16,
        bf16=caps.supports_bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        optim=common.pick_optimizer(caps, args.optim),
        seed=args.seed,
        report_to=args.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    if not args.train_on_full_sequence:
        markers = common.detect_response_markers(tokenizer, enable_thinking=args.enable_thinking)
        if markers is None:
            print("[提示] 未能识别对话模板的角色标记，本次对完整序列计算损失")
        else:
            from unsloth.chat_templates import train_on_responses_only

            instruction_part, response_part = markers
            print(f"[损失] 仅对助手回复计算损失：{instruction_part!r} -> {response_part!r}")
            trainer = train_on_responses_only(
                trainer,
                instruction_part=instruction_part,
                response_part=response_part,
            )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if not any((args.save_model, args.save_merged, args.save_gguf, args.push_model)):
        print("[保存] 未指定任何导出选项，训练结果只保留在 checkpoint 目录中")

    if args.save_model:
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"[保存] LoRA 适配器已写入 {args.save_path}")

    if args.save_merged:
        model.save_pretrained_merged(args.merged_path, tokenizer, save_method=args.save_method)
        print(f"[保存] 合并模型（{args.save_method}）已写入 {args.merged_path}")

    if args.save_gguf:
        print(f"正在导出 GGUF：{', '.join(args.quantization)}")
        result = model.save_pretrained_gguf(
            args.gguf_path,
            tokenizer,
            quantization_method=args.quantization,
        )
        if isinstance(result, dict):
            for path in result.get("gguf_files", []):
                print(f"GGUF 已保存到 {path}")
        else:
            print(f"GGUF 导出完成，请检查 {args.gguf_path}_gguf")

    if args.push_model:
        model.push_to_hub_merged(args.hub_path, tokenizer, token=args.hub_token, save_method=args.save_method)
        print(f"[推送] 已推送到 {args.hub_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 Unsloth 微调大语言模型")

    model_group = parser.add_argument_group("模型选项")
    model_group.add_argument("--model_name", type=str, default="models/Qwen3-1.7B",
                             help="本地模型目录或 HuggingFace 仓库名")
    model_group.add_argument("--max_seq_length", type=int, default=2048, help="最大序列长度，自动支持 RoPE 缩放")
    model_group.add_argument("--load_in_4bit", action="store_true", help="4bit 量化加载（需要可用的 bitsandbytes）")
    model_group.add_argument("--load_in_8bit", action="store_true", help="8bit 量化加载（需要可用的 bitsandbytes）")
    model_group.add_argument("--device_map", type=str, default="sequential",
                             help="设备分配；单卡用 sequential，多卡大模型可尝试 balanced")
    model_group.add_argument("--trust_remote_code", action="store_true",
                             help="仅对可信模型开启 Hub 远程代码执行")
    model_group.add_argument("--attn_implementation", type=str, default="auto",
                             choices=["auto", "flash_attention_2", "sdpa", "eager"],
                             help="auto 表示由 unsloth 自动选择，装了 flash-attn 会自动启用")
    model_group.add_argument("--disable_compile", action="store_true",
                             help="遇到 torch.compile / triton 编译报错时打开")

    data_group = parser.add_argument_group("数据选项")
    data_group.add_argument("--dataset", type=str, default="datasets/NetworkSecurity",
                            help="本地目录、json/jsonl/parquet 文件，或 HuggingFace 数据集名")
    data_group.add_argument("--dataset_split", type=str, default="train")
    data_group.add_argument("--data_files", type=str, nargs="+", default=None, help="只加载指定的数据文件")
    data_group.add_argument("--max_samples", type=int, default=None, help="只取前 N 条样本")
    data_group.add_argument("--system_prompt", type=str, default=None, help="统一写入的系统提示词")
    data_group.add_argument("--enable_thinking", action="store_true",
                            help="保留 Qwen3 系列的 <think> 段落进行训练")
    data_group.add_argument("--fallback_chat_template", type=str, default="chatml",
                            help="模型没有自带对话模板时使用的模板")
    data_group.add_argument("--train_on_full_sequence", action="store_true",
                            help="对提示词也计算损失（默认只对助手回复计算）")
    data_group.add_argument("--packing", action="store_true", help="把多条短样本打包进同一条序列")
    data_group.add_argument("--dataset_num_proc", type=int, default=1, help="Windows 上建议保持 1")

    lora_group = parser.add_argument_group("LoRA 选项")
    lora_group.add_argument("--r", type=int, default=16, help="LoRA 秩，常用 8/16/32/64")
    lora_group.add_argument("--lora_alpha", type=int, default=16)
    lora_group.add_argument("--lora_dropout", type=float, default=0.0,
                            help="默认 0 以使用 Unsloth 优化内核")
    lora_group.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"])
    lora_group.add_argument("--target_modules", type=str, nargs="+", default=None,
                            help="不指定则按模型实际结构自动检测")
    lora_group.add_argument("--use_gradient_checkpointing", type=str, default="unsloth")
    lora_group.add_argument("--random_state", type=int, default=3407)
    lora_group.add_argument("--use_rslora", action="store_true")
    lora_group.add_argument("--train_moe_experts", action="store_true", help="MoE 模型是否训练专家层")
    lora_group.add_argument("--moe_expert_layers", type=int, nargs="+", default=None,
                            help="只训练指定层号的专家，默认全部层")
    lora_group.add_argument("--moe_expert_rank", type=int, default=None,
                            help="融合专家的 LoRA 秩；默认按 r/专家数自动缩小")

    training_group = parser.add_argument_group("训练选项")
    training_group.add_argument("--per_device_train_batch_size", type=int, default=2)
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=4)
    training_group.add_argument("--warmup_steps", type=int, default=5)
    training_group.add_argument("--max_steps", type=int, default=60,
                                help="大于 0 时按步数训练，设为 -1 则按 epoch 训练完整数据集")
    training_group.add_argument("--num_train_epochs", type=float, default=1.0)
    training_group.add_argument("--learning_rate", type=float, default=2e-4)
    training_group.add_argument("--optim", type=str, default="adamw_8bit",
                                help="bitsandbytes 不可用时会自动降级为 adamw_torch_fused")
    training_group.add_argument("--weight_decay", type=float, default=0.01)
    training_group.add_argument("--lr_scheduler_type", type=str, default="linear")
    training_group.add_argument("--seed", type=int, default=3407)
    training_group.add_argument("--resume_from_checkpoint", nargs="?", const=True, default=False,
                                help="不带值时从最新 checkpoint 恢复，也可指定 checkpoint 路径")

    report_group = parser.add_argument_group("日志选项")
    report_group.add_argument("--report_to", type=str, default="tensorboard",
                              choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive",
                                       "flyte", "mlflow", "neptune", "tensorboard", "trackio", "wandb",
                                       "all", "none"])
    report_group.add_argument("--logging_steps", type=int, default=1)

    save_group = parser.add_argument_group("保存选项")
    save_group.add_argument("--output_dir", type=str, default="outputs", help="checkpoint 与训练日志目录")
    save_group.add_argument("--save_steps", type=int, default=20)
    save_group.add_argument("--save_model", action="store_true", help="训练结束后保存 LoRA 适配器")
    save_group.add_argument("--save_path", type=str, default="Unsloth-Models")
    save_group.add_argument("--save_merged", action="store_true", help="额外导出合并权重的完整模型")
    save_group.add_argument("--merged_path", type=str, default="Unsloth-Models-merged")
    save_group.add_argument("--save_method", type=str, default="merged_16bit",
                            choices=["merged_16bit", "merged_4bit", "lora"])
    save_group.add_argument("--save_gguf", action="store_true", help="导出 GGUF（需要本地可用的 llama.cpp）")
    save_group.add_argument("--gguf_path", type=str, default="Unsloth-Models-GGUF")
    save_group.add_argument("--quantization", type=str, default=["q4_k_m"], nargs="+",
                            help="GGUF 量化方式，可写多个，例如 f16 q8_0 q4_k_m")

    push_group = parser.add_argument_group("上传选项")
    push_group.add_argument("--push_model", action="store_true", help="训练结束后推送到 HuggingFace Hub")
    push_group.add_argument("--hub_path", type=str, default="hf/model")
    push_group.add_argument("--hub_token", type=str, default=None)

    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
