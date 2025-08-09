import sys
import unsloth
from transformers import TextStreamer
from unsloth import FastLanguageModel
from peft import PeftModel
import argparse


def main(model_name, load_in_4bit, question):
    max_seq_length = 2048  # 最大序列长度
    dtype = None  # 数据类型

    # 首先加载基础模型
    base_model_name = "models/Qwen3-1.7B"
    
    # 从预训练模型加载FastLanguageModel和分词器
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, model_name)

    # 将模型设置为推理模式
    FastLanguageModel.for_inference(model)

    # 对用户输入的问题进行分词，并转换为PyTorch张量，移动到CUDA设备
    inputs = tokenizer([question], return_tensors="pt").to("cuda")

    # 创建TextStreamer实例
    text_streamer = TextStreamer(tokenizer)

    # 生成模型的输出，并使用streamer实现流式输出
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1200,  # 生成的新token最大数量
        use_cache=True,  # 是否使用缓存
    )


if __name__ == "__main__":
    # 设置默认的模型名称
    default_model_name = "Unsloth-Qwen3"

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=default_model_name, help='The model name to use')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='Load the model in 4-bit precision if this flag is set')
    args = parser.parse_args()

    # 获取模型名称和是否以4位精度加载模型的选项
    model_name = args.model_name
    load_in_4bit = args.load_in_4bit

    # 获取用户输入的问题
    question = input("请输入您的问题：")

    # 使用模型名称、是否以4位精度加载模型的选项和问题调用主函数
    main(model_name, load_in_4bit, question)
