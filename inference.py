"""加载基础模型或 LoRA 适配器进行推理。"""

from __future__ import annotations

import argparse

# 必须先设置 UTF-8 和 Unsloth 环境。
import common

common.apply_runtime_env()

from unsloth import FastLanguageModel  # noqa: E402
from transformers import TextStreamer  # noqa: E402


def load_model(model_name: str, base_model: str | None, max_seq_length: int,
               load_in_4bit: bool, load_in_8bit: bool, attn_implementation: str,
               device_map: str, trust_remote_code: bool):
    caps = common.detect_capabilities()
    common.print_capabilities(caps)
    profile = common.inspect_model_profile(base_model or model_name, trust_remote_code=trust_remote_code)
    common.check_model_requirements(profile)

    load_in_4bit, load_in_8bit = common.pick_quantization(caps, load_in_4bit, load_in_8bit)
    common.warn_model_runtime(profile, caps, load_in_4bit, load_in_8bit)
    load_kwargs = dict(
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        load_in_16bit=not (load_in_4bit or load_in_8bit),
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        text_only=True,
        **common.resolve_attn_implementation(caps, attn_implementation),
    )

    if base_model:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            **load_kwargs,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_name)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            **load_kwargs,
        )

    tokenizer = common.ensure_chat_template(tokenizer, model_profile=profile)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens: int, temperature: float,
             top_p: float, enable_thinking: bool, max_seq_length: int) -> str:
    prompt = common.render_chat(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    max_input_tokens = max_seq_length - max_new_tokens
    if max_input_tokens <= 0:
        raise ValueError("max_new_tokens 必须小于 max_seq_length")

    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(model.device)
    finally:
        tokenizer.truncation_side = original_truncation_side

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        use_cache=True,
        pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
    )
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=top_p)
    outputs = model.generate(**generation_kwargs)
    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth LoRA 推理")
    parser.add_argument("--model_name", type=str, default="Unsloth-Models",
                        help="LoRA 适配器目录、合并后的模型目录，或基础模型名")
    parser.add_argument("--base_model", type=str, default=None,
                        help="适配器里记录的基础模型路径失效时，用它显式指定基础模型")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true", help="4bit 量化加载（需要可用的 bitsandbytes）")
    parser.add_argument("--load_in_8bit", action="store_true", help="8bit 量化加载（需要可用的 bitsandbytes）")
    parser.add_argument("--attn_implementation", type=str, default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--device_map", type=str, default="sequential")
    parser.add_argument("--trust_remote_code", action="store_true",
                        help="仅对可信模型开启 Hub 远程代码执行")
    parser.add_argument("--max_new_tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.7, help="设为 0 则使用贪心解码")
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--system", type=str, default=None, help="系统提示词")
    parser.add_argument("--thinking", action="store_true", help="打开 Qwen3 系列的思考模式")
    parser.add_argument("--question", type=str, default=None, help="单次提问，问完即退出")
    args = parser.parse_args()

    if args.max_seq_length <= 0:
        parser.error("--max_seq_length 必须大于 0")
    if args.max_new_tokens <= 0:
        parser.error("--max_new_tokens 必须大于 0")
    if args.max_new_tokens >= args.max_seq_length:
        parser.error("--max_new_tokens 必须小于 --max_seq_length")
    if args.temperature < 0:
        parser.error("--temperature 不能小于 0")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p 必须在 (0, 1] 范围内")

    model, tokenizer = load_model(
        args.model_name,
        args.base_model,
        args.max_seq_length,
        args.load_in_4bit,
        args.load_in_8bit,
        args.attn_implementation,
        args.device_map,
        args.trust_remote_code,
    )

    if args.thinking and not common.template_supports_thinking(tokenizer):
        print("[提示] 当前模型的对话模板不支持思考模式开关，--thinking 将被忽略")

    base_messages: list[dict[str, str]] = []
    if args.system:
        base_messages.append({"role": "system", "content": args.system})

    if args.question:
        generate(model, tokenizer, base_messages + [{"role": "user", "content": args.question}],
                 args.max_new_tokens, args.temperature, args.top_p, args.thinking, args.max_seq_length)
        return

    messages = list(base_messages)
    print("进入对话模式（/reset 清空上下文，/exit 退出）")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/reset":
            messages = list(base_messages)
            print("上下文已清空")
            continue

        messages.append({"role": "user", "content": question})
        print("助手：", end="", flush=True)
        answer = generate(model, tokenizer, messages, args.max_new_tokens,
                          args.temperature, args.top_p, args.thinking, args.max_seq_length)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
