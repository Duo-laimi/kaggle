import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = \
"""
<|im_start|>system
You are a good math problem solver. Give the final answer in 'Answer: \boxed{final_answer}' format.<|im_end|>
<|im_start|>user

{problem} 
<|im_end|>
<|im_start|>assistant
<think>

</think>


"""

class KaggleSolver:

    def __init__(
            self,
            model_path,
            dtype=None,
            max_seq_length=1024,
            inference_mode: bool = True,
            prompt_template: str = None
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.prompt_template = prompt_template or PROMPT_TEMPLATE

    def load(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
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

    @torch.no_grad()
    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self.model is None:
            self.load()
        prompt = self.prompt_template.format(problem=problem, final_answer="{final_answer}")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=self.max_seq_length)
        text = self.tokenizer.decode(output[0])
        if not self.inference_mode:
            print(f"Formatted Prompt: {prompt}")
            print(f"Generate Output: {text.split('</think>')[1]}")
        answer = self.extract_answer(text)
        return answer
