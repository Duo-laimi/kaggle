import re

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = \
"""
You are a good math problem solver.
Think and give the final answer to the problem.
"""

SYSTEM_PROMPT_formal = \
"""
You are a world-class International Mathematical Olympiad (IMO) competitor.
The final answer must be a non-negative integer between 0 and 99999.
You must place the final integer answer inside \\boxed{}.
"""

class KaggleSolver:

    def __init__(
            self,
            model_path,
            data_path,
            dtype=None,
            max_seq_length=1024,
            inference_mode: bool = True,
            system_prompt: str = None,
            **kwargs
    ):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.data_path = data_path
        self.model_kwargs = kwargs
        self.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05
        )
        self.dataset = None
        self.sft_config = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 1,
            # max_steps = 30,
            learning_rate = 2e-4,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none"
        )

    def load(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            device_map="auto",
            **self.model_kwargs
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

    def prepare_r1data(self, select=None):
        # parquet
        dataset = load_dataset(self.data_path, "default")["train"]
        unused_columns = ["uuid", "is_reasoning_complete", "generations", "correctness_math_verify",
                          "correctness_llama", "finish_reasons", "correctness_count", "messages"]
        dataset = dataset.remove_columns(unused_columns)
        if select is not None:
            dataset = dataset.select(select)

        def generate_conversation(examples):
            problems = examples["problem"]
            solutions = examples["solution"]
            answers = examples["answer"]
            conversations = []
            for problem, solution, answer in zip(problems, solutions, answers):
                conv = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": f"<think>{solution}</think>\n{answer}"}
                ]
                conversations.append(conv)

            return {"conversations": conversations}
        dataset_conv = self.tokenizer.apply_chat_template(
            list(dataset.map(generate_conversation, batched=True)["conversations"]),
            tokenize=False
        )
        self.dataset = dataset_conv

    def train(self):
        if self.model is None:
            self.load()
        if self.dataset is None:
            self.prepare_r1data()
        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=self.sft_config,
            peft_config=self.peft_config,
        )
        sft_trainer.train()


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


    from transformers import Trainer