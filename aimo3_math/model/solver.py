import re
from typing import Any, Dict
from openai import OpenAI

from datasets import Dataset
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from .prompt import system_prompt, tool_prompt, preference_prompt

class KaggleSolver:

    def __init__(
            self,
            model_path: str,
            dataset: Dataset = None,
            base_url: str = None,
            api_key: str = None,
            max_turns: int = 128,
            enable_thinking: bool = False,
            max_context_length: int = 4096,
            train_before_inference: bool = True,
            use_peft_train: bool = True,
            peft_weight_save_path: str = "peft_adapter",
            quantize_base_model: bool = True,
            peft_config: Dict[str, Any] = None,
            sft_config: Dict[str, Any] = None,
            quantization_config: Dict[str, Any] = None,
            **kwargs
    ):
        self.model_path = model_path
        self.model = None
        self.dataset = dataset
        self.offline_mode = True
        if base_url is not None:
            self.offline_mode = False
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_turns = max_turns
        self.enable_thinking = enable_thinking
        self.max_context_length = max_context_length
        self.train_before_inference = train_before_inference
        self.use_peft_train = use_peft_train
        self.peft_weight_save_path = peft_weight_save_path
        self.quantize_base_model = quantize_base_model
        self.peft_config = peft_config
        self.sft_config = sft_config
        self.quantization_config = quantization_config
        self.model_kwargs = kwargs
        self.system_prompt = f"{system_prompt}\n{tool_prompt}"

    def load(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quantization_config,
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model.config.use_cache = True
        # self.collator =
        print(f"Successfully load model from {self.model_path}.")

    def extract_answer(self, text):
        # 1. 移除思考过程，获取输出部分
        if "</think>" in text:
            _, output = text.split("</think>", 1)
        else:
            output = text

        # 2. 匹配 \boxed{内容} 中的内容
        # 使用捕获组 () 来提取数字，并处理可能存在的空格
        pattern = r"\\boxed{([^{}]+)}"
        matches = re.findall(pattern, output)
        # 3. 返回最后一个匹配项
        if matches:
            return matches[-1].strip()
        return None

    def train(self):
        if self.model is None:
            self.load()
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        self.model = prepare_model_for_kbit_training(self.model)
        # self.model.enable_input_require_grads()
        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=self.sft_config,
            peft_config=self.peft_config,
        )
        print("Training Start...")
        sft_trainer.train()
        print("Done!")
        if self.peft_weight_save_path is not None:
            sft_trainer.save_model(self.peft_weight_save_path)
        self.model.eval()
        self.model.config.use_cache = True

    def generate(self, conv):
        if self.offline_mode:
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_length=self.max_context_length)
            text = self.tokenizer.decode(output[0])
            # 剔除思考
            if "</think>" in text:
                _, text = text.split("</think>", 1)
        else:
            response = self.client.chat.completions.create(
                messages=conv,
                model=self.model_path,
                max_tokens=self.max_context_length,
                extra_body={"enable_thinking": self.enable_thinking},
            )
            text = response.choices[0].message.content
        return text

    # 工具调用：python
    # 多轮对话：会话管理

    def predict(self, problem: str):
        if self.model is None:
            self.load()
        if self.train_before_inference:
            self.train()
            self.train_before_inference = False

        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem}\n{preference_prompt}"},
        ]

        for t in range(self.max_turns):
            pass

        text = self.generate(conv)
        answer = self.extract_answer(text)
        try:
            answer = int(answer)
        except Exception as e:
            answer = 0
        return answer