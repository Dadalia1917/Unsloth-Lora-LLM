"""训练和推理脚本共用的兼容函数。"""

from __future__ import annotations

import os
import re
import sys
import glob
import json
import hashlib
import subprocess
import importlib.util
import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


_BOOTSTRAP_FLAG = "UNSLOTH_LORA_UTF8_BOOTSTRAP"


def ensure_utf8_mode() -> None:
    """Windows 非 UTF-8 环境下用 UTF-8 模式重新启动脚本。"""
    if sys.flags.utf8_mode or os.environ.get(_BOOTSTRAP_FLAG):
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        return

    entry = sys.argv[0] if sys.argv else ""
    if not entry.endswith(".py") or not os.path.exists(entry):
        print("[警告] 当前解释器不在 UTF-8 模式，请先设置环境变量 PYTHONUTF8=1，否则 import unsloth 可能失败")
        return

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env[_BOOTSTRAP_FLAG] = "1"

    command = [sys.executable, "-X", "utf8", *sys.argv]
    raise SystemExit(subprocess.run(command, env=env).returncode)


def apply_runtime_env(compile_disable: bool = False) -> None:
    """设置 Unsloth 导入前需要的环境变量。"""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if compile_disable:
        os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"


ensure_utf8_mode()


@dataclass
class RuntimeCapabilities:
    """当前环境可用的训练能力。"""

    torch_version: str = ""
    cuda_version: str | None = None
    compute_capability: tuple[int, int] | None = None
    unsloth_version: str = ""
    unsloth_zoo_version: str = ""
    transformers_version: str = ""
    trl_version: str = ""
    peft_version: str = ""
    datasets_version: str = ""
    bitsandbytes_version: str = ""
    gpu_name: str = ""
    vram_gb: float = 0.0
    supports_bf16: bool = False
    has_bitsandbytes: bool = False
    has_flash_attn: bool = False
    flash_attn_version: str = ""
    has_causal_conv1d: bool = False
    has_fla: bool = False
    notes: list[str] = field(default_factory=list)


def _bitsandbytes_usable() -> tuple[bool, str]:
    """检查 bitsandbytes 的原生库和优化器是否可用。"""
    try:
        import bitsandbytes as bnb
    except Exception as exc:  # noqa: BLE001
        return False, f"bitsandbytes 不可用：{exc}"

    try:
        from bitsandbytes import cextension
    except Exception:  # noqa: BLE001
        cextension = None

    if cextension is not None and getattr(cextension, "lib", None) is None:
        return False, "bitsandbytes 原生库未加载（缺少匹配当前 CUDA 的 DLL），4bit/8bit 量化与 8bit 优化器不可用"
    try:
        bnb.optim.AdamW8bit  # noqa: B018 - 仅确认符号存在
    except Exception as exc:  # noqa: BLE001
        return False, f"bitsandbytes 优化器不可用：{exc}"
    return True, ""


def detect_capabilities() -> RuntimeCapabilities:
    import torch

    caps = RuntimeCapabilities()
    caps.torch_version = torch.__version__
    caps.cuda_version = torch.version.cuda

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return ""

    caps.unsloth_version = package_version("unsloth")
    caps.unsloth_zoo_version = package_version("unsloth_zoo")
    caps.peft_version = package_version("peft")
    caps.datasets_version = package_version("datasets")
    caps.bitsandbytes_version = package_version("bitsandbytes")

    try:
        import transformers

        caps.transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import trl

        caps.trl_version = trl.__version__
    except Exception:  # noqa: BLE001
        pass

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        caps.gpu_name = props.name
        caps.vram_gb = round(props.total_memory / 1024 ** 3, 1)
        caps.compute_capability = torch.cuda.get_device_capability(0)
        try:
            caps.supports_bf16 = torch.cuda.is_bf16_supported()
        except Exception:  # noqa: BLE001
            caps.supports_bf16 = props.major >= 8
    else:
        caps.notes.append("未检测到可用的 CUDA 设备，LoRA 微调需要 NVIDIA GPU")

    usable, reason = _bitsandbytes_usable()
    caps.has_bitsandbytes = usable
    if reason:
        caps.notes.append(reason)

    try:
        import flash_attn

        caps.has_flash_attn = True
        caps.flash_attn_version = getattr(flash_attn, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        caps.notes.append("未安装可用的 flash-attn，将回退到 xformers / SDPA 注意力实现")

    caps.has_causal_conv1d = importlib.util.find_spec("causal_conv1d") is not None
    caps.has_fla = importlib.util.find_spec("fla") is not None

    # 自定义 torch wheel 可能能正常导入，但仍超出 Unsloth 声明的版本范围。
    try:
        from packaging.requirements import Requirement

        for distribution in ("unsloth", "unsloth_zoo"):
            for raw_requirement in importlib.metadata.requires(distribution) or ():
                requirement = Requirement(raw_requirement)
                if requirement.name.lower().replace("-", "_") != "torch":
                    continue
                if requirement.marker and not requirement.marker.evaluate():
                    continue
                if requirement.specifier and not requirement.specifier.contains(torch.__version__, prereleases=True):
                    caps.notes.append(
                        f"{distribution} 的包元数据要求 torch{requirement.specifier}，"
                        f"当前 {torch.__version__} 属于自定义、非官方声明组合"
                    )
    except Exception:  # noqa: BLE001
        pass

    return caps


def print_capabilities(caps: RuntimeCapabilities) -> None:
    print("运行环境：")
    print(f"  torch            : {caps.torch_version} (CUDA {caps.cuda_version})")
    print(f"  unsloth/zoo      : {caps.unsloth_version} / {caps.unsloth_zoo_version}")
    print(f"  transformers/trl : {caps.transformers_version} / {caps.trl_version}")
    print(f"  peft/datasets    : {caps.peft_version} / {caps.datasets_version}")
    capability = ".".join(map(str, caps.compute_capability)) if caps.compute_capability else "-"
    print(f"  GPU              : {caps.gpu_name or '无'}  显存 {caps.vram_gb} GB  SM {capability}")
    print(f"  bf16             : {'支持' if caps.supports_bf16 else '不支持（回退 fp16）'}")
    bnb_status = caps.bitsandbytes_version if caps.has_bitsandbytes else "不可用"
    print(f"  bitsandbytes     : {bnb_status}")
    print(f"  flash-attn       : {caps.flash_attn_version if caps.has_flash_attn else '不可用'}")
    print(
        "  linear-attn 加速 : "
        f"causal-conv1d={'有' if caps.has_causal_conv1d else '无'} / FLA={'有' if caps.has_fla else '无'}"
    )
    for note in caps.notes:
        print(f"  注意             : {note}")


def pick_optimizer(caps: RuntimeCapabilities, preferred: str = "adamw_8bit") -> str:
    """bitsandbytes 不可用时改用 torch 优化器。"""
    if preferred.endswith("8bit") and not caps.has_bitsandbytes:
        print(f"[降级] optim={preferred} 需要 bitsandbytes，实际不可用，改用 adamw_torch_fused")
        return "adamw_torch_fused"
    return preferred


def resolve_attn_implementation(caps: RuntimeCapabilities, requested: str = "auto") -> dict[str, str]:
    """返回模型加载所需的注意力参数。"""
    if not requested or requested == "auto":
        return {}
    if requested == "flash_attention_2" and not caps.has_flash_attn:
        print("[降级] 指定了 flash_attention_2 但 flash-attn 不可用，改由 unsloth 自动选择注意力实现")
        return {}
    return {"attn_implementation": requested}


def pick_quantization(caps: RuntimeCapabilities, load_in_4bit: bool, load_in_8bit: bool) -> tuple[bool, bool]:
    """检查量化配置，返回最终的 4bit 和 8bit 开关。"""
    if (load_in_4bit or load_in_8bit) and not caps.has_bitsandbytes:
        print("[降级] 请求了量化加载但 bitsandbytes 不可用，改为 bf16/fp16 全精度权重 + LoRA")
        return False, False
    if load_in_4bit and load_in_8bit:
        print("[降级] load_in_4bit 与 load_in_8bit 互斥，保留 4bit")
        return True, False
    return load_in_4bit, load_in_8bit


@dataclass
class ModelProfile:
    """从 config.json 读取的模型信息。"""

    requested_name: str
    base_model_name: str
    model_type: str = ""
    architectures: tuple[str, ...] = ()
    display_name: str = "未知模型"
    is_moe: bool = False
    uses_linear_attention: bool = False
    config: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_qwen35_family(self) -> bool:
        return self.model_type in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}

    @property
    def is_qwen36(self) -> bool:
        names = f"{self.requested_name} {self.base_model_name}"
        return bool(re.search(r"qwen[-_ ]?3[._-]6", names, re.I))

    @property
    def is_deepseek_v4(self) -> bool:
        names = f"{self.requested_name} {self.base_model_name} {self.model_type}"
        return self.model_type == "deepseek_v4" or bool(re.search(r"deepseek[-_ ]?v4", names, re.I))


def resolve_adapter_base_model(model_name: str) -> str:
    """从 LoRA 配置中解析基础模型。"""
    adapter_config = os.path.join(model_name, "adapter_config.json")
    if not os.path.isfile(adapter_config):
        return model_name
    try:
        with open(adapter_config, "r", encoding="utf-8") as file:
            base_model = json.load(file).get("base_model_name_or_path")
    except (OSError, ValueError):
        return model_name
    if not base_model:
        return model_name
    if os.path.isabs(base_model) or os.path.exists(base_model):
        return base_model

    # 训练时记录的相对路径可能需要从适配器目录的父目录解析。
    candidate = os.path.normpath(os.path.join(os.path.abspath(model_name), os.pardir, base_model))
    return candidate if os.path.exists(candidate) else base_model


def _read_raw_model_config(model_name: str, trust_remote_code: bool = False) -> dict[str, Any]:
    config_path = os.path.join(model_name, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    from transformers import PreTrainedConfig

    config, _ = PreTrainedConfig.get_config_dict(model_name, trust_remote_code=trust_remote_code)
    return config


def inspect_model_profile(model_name: str, trust_remote_code: bool = False) -> ModelProfile:
    """读取本地或 Hub 的模型配置。"""
    base_model_name = resolve_adapter_base_model(model_name)
    try:
        config = _read_raw_model_config(base_model_name, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 - 真正的模型加载会给出完整的 Hub / 权限错误
        print(f"[模型] 暂时无法读取 config.json，将按模型名判断兼容性：{exc}")
        config = {}

    text_config = config.get("text_config") if isinstance(config.get("text_config"), Mapping) else config
    model_type = str(config.get("model_type") or text_config.get("model_type") or "")
    architectures = tuple(str(item) for item in (config.get("architectures") or ()))
    combined_name = f"{model_name} {base_model_name} {model_type} {' '.join(architectures)}"

    moe_keys = ("num_experts", "n_routed_experts", "num_local_experts", "moe_intermediate_size")
    is_moe = any(text_config.get(key) for key in moe_keys)
    layer_types = text_config.get("layer_types") or config.get("layer_types") or ()
    uses_linear_attention = any("linear" in str(layer_type).lower() for layer_type in layer_types)

    if model_type == "deepseek_v4" or re.search(r"deepseek[-_ ]?v4", combined_name, re.I):
        display_name = "DeepSeek-V4"
    elif re.search(r"qwen[-_ ]?3[._-]6", combined_name, re.I):
        display_name = "Qwen3.6"
    elif model_type.startswith("qwen3_5") or re.search(r"qwen[-_ ]?3[._-]5", combined_name, re.I):
        display_name = "Qwen3.5"
    elif model_type.startswith("qwen3"):
        display_name = "Qwen3"
    elif model_type:
        display_name = model_type
    else:
        display_name = os.path.basename(os.path.normpath(base_model_name)) or model_name

    return ModelProfile(
        requested_name=model_name,
        base_model_name=base_model_name,
        model_type=model_type,
        architectures=architectures,
        display_name=display_name,
        is_moe=is_moe,
        uses_linear_attention=uses_linear_attention,
        config=config,
    )


def check_model_requirements(profile: ModelProfile | str) -> ModelProfile:
    """加载权重前检查模型架构和 Transformers 版本。"""
    if isinstance(profile, str):
        profile = inspect_model_profile(profile)

    import transformers
    from packaging.version import Version
    from transformers import CONFIG_MAPPING

    current = Version(transformers.__version__)
    if profile.is_deepseek_v4:
        if current < Version("5.9.0") or "deepseek_v4" not in CONFIG_MAPPING:
            raise RuntimeError(
                "DeepSeek-V4 需要 transformers>=5.9.0 且包含 deepseek_v4 架构，"
                f"当前版本为 {transformers.__version__}。请使用彼此兼容的 Transformers、"
                "Unsloth、TRL 和 PEFT 版本，不要只单独升级其中一个包。"
            )
    elif profile.is_qwen36 and current < Version("5.5.0"):
        raise RuntimeError(f"Qwen3.6 需要 transformers>=5.5.0，当前为 {transformers.__version__}")
    elif profile.is_qwen35_family and current < Version("5.2.0"):
        raise RuntimeError(f"Qwen3.5 需要 transformers v5（本项目最低 5.2.0），当前为 {transformers.__version__}")
    elif re.search(r"gemma[-_ ]?4", f"{profile.requested_name} {profile.model_type}", re.I) and current < Version("5.0.0"):
        raise RuntimeError(f"Gemma 4 需要 transformers>=5.0.0，当前为 {transformers.__version__}")

    return profile


def warn_model_runtime(profile: ModelProfile, caps: RuntimeCapabilities,
                       load_in_4bit: bool = False, load_in_8bit: bool = False) -> None:
    """打印模型相关的性能和精度提示。"""
    print(
        f"[模型] {profile.display_name}  model_type={profile.model_type or '未知'}"
        f"  架构={','.join(profile.architectures) or '未知'}"
    )
    if profile.uses_linear_attention and not (caps.has_causal_conv1d and caps.has_fla):
        print(
            "[提示] 检测到混合线性注意力。causal-conv1d / FLA 是 Transformers 参考实现的可选加速，"
            "当前未全部安装；Unsloth 会优先使用自己的 Triton 内核，不能使用时会回退但速度更慢、显存更高。"
        )
    if profile.is_qwen35_family and (load_in_4bit or load_in_8bit):
        print("[警告] Unsloth 不建议 Qwen3.5/3.6 使用量化 LoRA；为避免较大精度损失，优先使用 bf16 LoRA")
    if profile.is_moe and (load_in_4bit or load_in_8bit):
        print("[警告] 当前是 MoE 模型；bitsandbytes 对融合专家量化支持有限，bf16 LoRA 通常更稳定")


def is_moe_model(model: Any) -> bool:
    config = getattr(model, "config", None)
    if config is None:
        return False
    text_config = getattr(config, "text_config", config)
    moe_keys = (
        "num_experts",
        "n_routed_experts",
        "num_local_experts",
        "moe_intermediate_size",
        "num_experts_per_tok",
    )
    return any(getattr(text_config, key, None) for key in moe_keys)


# 不对输出头、嵌入和 MoE 路由器注入 LoRA。
_EXCLUDED_LEAF_NAMES = {"lm_head", "gate", "router", "shared_expert_gate", "wg", "e_score_correction_bias"}
_EXCLUDED_PATH_PARTS = {
    "embed_tokens", "embeddings", "rotary_emb", "norm", "lm_head", "output_head",
    "router", "shared_expert_gate", "vision_tower", "visual", "vision_model",
    "multi_modal_projector", "mm_projector",
}

_PREFERRED_ORDER = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "qkv_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "wqkv",
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_a_proj", "o_b_proj",
    "gate_proj", "up_proj", "down_proj",
    "gate_up_proj", "w1", "w2", "w3",
    "fc1", "fc2", "dense", "dense_h_to_4h", "dense_4h_to_h",
]


def auto_lora_target_modules(model: Any, include_moe_experts: bool = False,
                             text_only: bool = True) -> list[str]:
    """扫描模型中适合注入 LoRA 的线性层。"""
    found: set[str] = set()
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        weight = getattr(module, "weight", None)
        if "Linear" not in class_name and class_name != "Conv1D":
            continue
        if weight is not None and getattr(weight, "ndim", 2) != 2:
            continue
        if not include_moe_experts and ".experts." in f".{name}.":
            continue
        path_parts = set(name.lower().split("."))
        if text_only and path_parts & {"vision_tower", "visual", "vision_model", "multi_modal_projector", "mm_projector"}:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if not leaf or leaf.isdigit():
            continue
        if leaf in _EXCLUDED_LEAF_NAMES:
            continue
        if path_parts & _EXCLUDED_PATH_PARTS:
            continue
        found.add(leaf)

    ordered = [name for name in _PREFERRED_ORDER if name in found]
    ordered += sorted(found - set(ordered))
    return ordered


def auto_moe_target_parameters(model: Any, layers: Sequence[int] | None = None) -> list[str]:
    """返回 MoE 的三维融合专家参数。"""
    names: list[str] = []
    for name, param in model.named_parameters():
        if param.ndim != 3 or ".experts." not in f".{name}.":
            continue
        if layers is not None:
            match = re.search(r"layers\.(\d+)\.", name)
            if match is None or int(match.group(1)) not in layers:
                continue
        names.append(name)
    return sorted(names)


def moe_expert_rank_pattern(model: Any, target_parameters: Sequence[str], default_r: int) -> dict[str, int]:
    """按专家数缩小融合专家的 LoRA rank。"""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    num_experts = 0
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        num_experts = int(getattr(text_config, key, 0) or 0)
        if num_experts:
            break
    # 部分模型未在 config 中记录专家数，可从融合权重第一维推断。
    if not num_experts:
        parameters = dict(model.named_parameters())
        for name in target_parameters:
            parameter = parameters.get(name)
            if parameter is not None and parameter.ndim == 3:
                num_experts = int(parameter.shape[0])
                break
    if not num_experts:
        return {}
    expert_r = max(1, default_r // num_experts)
    return {name: expert_r for name in target_parameters}


_PROBE_USER = "\u0001USER_PROBE\u0001"
_PROBE_ASSISTANT = "\u0001ASSISTANT_PROBE\u0001"
_SPECIAL_TOKEN_PATTERN = re.compile(
    r"<\|[^|<>]*\|>|<｜[^｜<>]*｜>|</?[a-zA-Z_][a-zA-Z0-9_]*>|\[/?INST\]|\[/?SYSTEM_PROMPT\]"
)

# DeepSeek-V4 没有 Jinja 模板，这里只实现纯文本 SFT 所需的基础格式。
_DEEPSEEK_V4_BASIC_CHAT_TEMPLATE = r"""
{{- '<｜begin▁of▁sentence｜>' -}}
{%- for message in messages -%}
    {%- if message['role'] == 'system' -%}
        {{- message['content'] -}}
    {%- elif message['role'] == 'user' -%}
        {{- '<｜User｜>' + message['content'] -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- '<｜Assistant｜>' -}}
        {%- if message['content'].startswith('<think>') -%}
            {{- message['content'] -}}
        {%- elif enable_thinking is defined and enable_thinking -%}
            {{- '<think>\n\n</think>\n\n' + message['content'] -}}
        {%- else -%}
            {{- '</think>' + message['content'] -}}
        {%- endif -%}
        {{- '<｜end▁of▁sentence｜>' -}}
    {%- else -%}
        {{- raise_exception('DeepSeek-V4 基础模板只支持 system/user/assistant；工具调用请使用官方 encoding_dsv4.py') -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<｜Assistant｜>' -}}
    {%- if enable_thinking is defined and enable_thinking -%}
        {{- '<think>' -}}
    {%- else -%}
        {{- '</think>' -}}
    {%- endif -%}
{%- endif -%}
""".strip()


def ensure_chat_template(tokenizer: Any, fallback_template: str = "chatml",
                         model_profile: ModelProfile | None = None) -> Any:
    """补全缺失的 chat template。"""
    if getattr(tokenizer, "chat_template", None):
        if model_profile and model_profile.is_deepseek_v4:
            tokenizer._unsloth_lora_template_family = "deepseek_v4"
        return tokenizer

    if model_profile and model_profile.is_deepseek_v4:
        print("[模板] DeepSeek-V4 没有 Jinja 模板，套用官方 encoding 格式的纯文本 SFT 子集")
        tokenizer.chat_template = _DEEPSEEK_V4_BASIC_CHAT_TEMPLATE
        tokenizer._unsloth_lora_template_family = "deepseek_v4"
        return tokenizer

    from unsloth.chat_templates import get_chat_template

    print(f"[模板] 分词器没有内置 chat_template，套用 unsloth 的 {fallback_template} 模板")
    return get_chat_template(tokenizer, chat_template=fallback_template)


def template_supports_thinking(tokenizer: Any) -> bool:
    """检查模板是否支持 enable_thinking。"""
    template = getattr(tokenizer, "chat_template", "") or ""
    if isinstance(template, Mapping):
        return any("enable_thinking" in str(value) for value in template.values())
    return "enable_thinking" in str(template)


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    enable_thinking: bool | None = None,
) -> str:
    """使用模型的 chat template 渲染对话。"""
    kwargs: dict[str, Any] = {}
    if enable_thinking is not None and template_supports_thinking(tokenizer):
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def detect_response_markers(tokenizer: Any, enable_thinking: bool = False) -> tuple[str, str] | None:
    """识别 train_on_responses_only 需要的角色标记。"""
    if getattr(tokenizer, "_unsloth_lora_template_family", None) == "deepseek_v4":
        response = "<｜Assistant｜><think>" if enable_thinking else "<｜Assistant｜></think>"
        return "<｜User｜>", response

    try:
        effective_thinking = enable_thinking and template_supports_thinking(tokenizer)
        assistant_probe = f"<think>\n{_PROBE_ASSISTANT}" if effective_thinking else _PROBE_ASSISTANT
        rendered = render_chat(
            tokenizer,
            [
                {"role": "user", "content": _PROBE_USER},
                {"role": "assistant", "content": assistant_probe},
            ],
            enable_thinking=effective_thinking,
        )
    except Exception:  # noqa: BLE001
        return None

    matches = list(_SPECIAL_TOKEN_PATTERN.finditer(rendered))

    def marker_before(probe: str) -> str | None:
        index = rendered.find(probe)
        if index < 0:
            return None
        before = [match for match in matches if match.end() <= index]
        if not before:
            return None

        start = before[-1].start()
        # Llama3 的角色名夹在两个特殊 token 之间，需要把整段角色头一起返回。
        for match in reversed(before[:-1]):
            gap = rendered[match.end():start]
            if gap and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]{0,19}", gap):
                start = match.start()
            else:
                break
        return rendered[start:index]

    instruction_part = marker_before(_PROBE_USER)
    response_part = marker_before(_PROBE_ASSISTANT)
    if not instruction_part or not response_part or instruction_part == response_part:
        return None
    return instruction_part, response_part


_ROLE_ALIASES = {
    "human": "user", "user": "user", "prompter": "user", "instruction": "user",
    "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "chatgpt": "assistant", "model": "assistant",
    "system": "system", "tool": "tool", "function": "tool", "observation": "tool",
}
_CONVERSATION_KEYS = ("messages", "conversations", "conversation", "chat", "dialog")
_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query", "input_text")
_INPUT_KEYS = ("input", "context", "reference")
_OUTPUT_KEYS = ("output", "answer", "response", "completion", "chosen", "content")
_THINKING_KEYS = ("think", "thinking", "reasoning", "reasoning_content", "chain_of_thought", "cot", "complex_cot")

_DATA_SUFFIXES = ("json", "jsonl", "csv", "parquet", "txt")


def ensure_datasets_python314_compat() -> None:
    """兼容 Python 3.14 与 datasets 4.3 的 Pickler 签名。"""
    if sys.version_info < (3, 14):
        return

    import datasets
    from packaging.version import Version

    if Version(datasets.__version__) >= Version("4.4.0"):
        return
    from datasets.utils import _dill as datasets_dill

    current = datasets_dill.Pickler._batch_setitems
    if getattr(current, "_unsloth_lora_py314_compat", False):
        return

    def _batch_setitems(self, items, *args, **kwargs):
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            return datasets_dill.dill.Pickler._batch_setitems(self, items, *args, **kwargs)
        try:
            items = sorted(items)
        except Exception:  # TypeError, decimal.InvalidOperation, etc.
            from datasets.fingerprint import Hasher

            items = sorted(items, key=lambda item: Hasher.hash(item[0]))
        return datasets_dill.dill.Pickler._batch_setitems(self, items, *args, **kwargs)

    _batch_setitems._unsloth_lora_py314_compat = True
    datasets_dill.Pickler._batch_setitems = _batch_setitems
    print("[兼容] 已应用 datasets 4.3 + Python 3.14 的 Pickler 签名修复")


def _dataset_builder_for(path: str) -> str:
    suffix = path.rsplit(".", 1)[-1].lower()
    if suffix in ("json", "jsonl"):
        return "json"
    if suffix == "txt":
        return "text"
    return suffix


def load_any_dataset(source: str, split: str = "train", data_files: str | list[str] | None = None):
    """加载本地文件、磁盘数据集或 Hugging Face 数据集。"""
    ensure_datasets_python314_compat()
    from datasets import DatasetDict, load_dataset, load_from_disk

    if data_files is not None:
        first_file = data_files[0] if isinstance(data_files, list) else data_files
        builder = _dataset_builder_for(first_file)
        return load_dataset(builder, data_files=data_files, split=split)

    if os.path.isdir(source):
        if os.path.exists(os.path.join(source, "dataset_info.json")) or os.path.exists(
            os.path.join(source, "state.json")
        ):
            dataset = load_from_disk(source)
            if isinstance(dataset, DatasetDict):
                return dataset[split] if split in dataset else dataset[next(iter(dataset))]
            return dataset

        for suffix in _DATA_SUFFIXES:
            files = sorted(glob.glob(os.path.join(source, f"**/*.{suffix}"), recursive=True))
            if files:
                builder = _dataset_builder_for(files[0])
                return load_dataset(builder, data_files=files, split=split)

        return load_dataset(source, split=split)

    if os.path.isfile(source):
        builder = _dataset_builder_for(source)
        return load_dataset(builder, data_files=source, split=split)

    return load_dataset(source, split=split)


def _first_key(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    columns = list(columns)
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return "" if content is None else str(content)


def _normalize_conversation(raw: Any, include_reasoning: bool = False) -> list[dict[str, str]]:
    """统一 ShareGPT 和 OpenAI 对话格式。"""
    messages: list[dict[str, str]] = []
    if not isinstance(raw, (list, tuple)):
        return messages
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker") or ""
        content = turn.get("content")
        if content is None:
            content = turn.get("value") or turn.get("text") or ""
        role = _ROLE_ALIASES.get(str(role).strip().lower(), "user")
        content = _content_to_text(content)
        if role == "assistant" and include_reasoning:
            reasoning = ""
            for key in _THINKING_KEYS:
                if turn.get(key):
                    reasoning = _content_to_text(turn[key]).strip()
                    break
            if reasoning and not content.lstrip().startswith("<think>"):
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        messages.append({"role": role, "content": content})
    return messages


def build_text_dataset(
    dataset,
    tokenizer: Any,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
    keep_dataset_thinking: bool | None = None,
    num_proc: int = 1,
    text_field: str = "text",
):
    """把常见数据集格式转换成模型可训练的 text 列。"""
    columns = list(dataset.column_names or [])
    if text_field in columns:
        return dataset

    conversation_key = _first_key(columns, _CONVERSATION_KEYS)
    instruction_key = _first_key(columns, _INSTRUCTION_KEYS)
    input_key = _first_key(columns, _INPUT_KEYS)
    output_key = _first_key(columns, _OUTPUT_KEYS)
    thinking_key = _first_key(columns, _THINKING_KEYS)

    if conversation_key is None and instruction_key is None and input_key is not None:
        instruction_key, input_key = input_key, None

    if conversation_key is None and (instruction_key is None or output_key is None):
        raise ValueError(
            f"无法识别数据集字段：{columns}。请提供 text 列、对话列（messages/conversations），"
            f"或 Alpaca 风格的 instruction/input/output 列。"
        )

    supports_thinking = template_supports_thinking(tokenizer)
    include_reasoning = enable_thinking if keep_dataset_thinking is None else keep_dataset_thinking

    def to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
        if conversation_key is not None:
            messages = _normalize_conversation(example[conversation_key], include_reasoning=include_reasoning)
        else:
            question = _content_to_text(example.get(instruction_key))
            extra = _content_to_text(example.get(input_key)) if input_key else ""
            if extra.strip():
                question = f"{question}\n\n{extra}"
            answer = _content_to_text(example.get(output_key))
            if thinking_key and include_reasoning:
                thought = _content_to_text(example.get(thinking_key)).strip()
                if thought and supports_thinking:
                    answer = f"<think>\n{thought}\n</think>\n\n{answer}"
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        if not messages:
            raise ValueError("发现空对话样本，请检查 messages/conversations 字段")
        if system_prompt and not any(message["role"] == "system" for message in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def formatting(examples: dict[str, list[Any]]) -> dict[str, list[str]]:
        size = len(next(iter(examples.values())))
        texts: list[str] = []
        for index in range(size):
            example = {key: values[index] for key, values in examples.items()}
            texts.append(
                render_chat(
                    tokenizer,
                    to_messages(example),
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
            )
        return {text_field: texts}

    fingerprint_payload = json.dumps(
        {
            "version": 2,
            "source": getattr(dataset, "_fingerprint", ""),
            "columns": columns,
            "chat_template": getattr(tokenizer, "chat_template", "") or "",
            "system_prompt": system_prompt,
            "enable_thinking": enable_thinking,
            "include_reasoning": include_reasoning,
            "text_field": text_field,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    new_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()

    return dataset.map(
        formatting,
        batched=True,
        num_proc=num_proc if num_proc and num_proc > 1 else None,
        remove_columns=columns,
        new_fingerprint=new_fingerprint,
        desc="套用对话模板",
    )


__all__ = [
    "ModelProfile",
    "RuntimeCapabilities",
    "apply_runtime_env",
    "auto_lora_target_modules",
    "auto_moe_target_parameters",
    "build_text_dataset",
    "check_model_requirements",
    "detect_capabilities",
    "detect_response_markers",
    "ensure_chat_template",
    "ensure_datasets_python314_compat",
    "ensure_utf8_mode",
    "is_moe_model",
    "inspect_model_profile",
    "load_any_dataset",
    "moe_expert_rank_pattern",
    "pick_optimizer",
    "pick_quantization",
    "print_capabilities",
    "render_chat",
    "resolve_adapter_base_model",
    "resolve_attn_implementation",
    "template_supports_thinking",
    "warn_model_runtime",
]
