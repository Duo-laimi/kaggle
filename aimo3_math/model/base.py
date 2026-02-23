import re

import torch
import pandas as pd
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, PeftModel, prepare_model_for_kbit_training
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
            dtype: torch.dtype = None,
            select: int = None,
            max_seq_length: int = 1024,
            inference_mode: bool = True,
            system_prompt: str = None,
            train_before_inference: bool = True,
            from_peft_pretrained: bool = False,
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
        self.select = select
        self.sft_config = SFTConfig(
            dataset_text_field="text",
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
        self.from_peft_pretrained = from_peft_pretrained
        self.train_before_inference = train_before_inference

    def load(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_4bit=True,
            device_map="auto",
            **self.model_kwargs
        )
        if self.from_peft_pretrained:
            self.model = PeftModel.from_pretrained(self.model, "lora_adapter")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        print(f"Successfully load model from {self.model_path}.")

    def train(self):
        if self.model is None:
            self.load()
        if self.dataset is None:
            self.dataset = load_dataset("json", data_files=self.data_path)
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)
        # self.model.enable_input_require_grads()
        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,
            args=self.sft_config,
            peft_config=self.peft_config,
        )
        sft_trainer.train()
        sft_trainer.save_model("lora_adapter")


    @torch.no_grad()
    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self.model is None:
            self.load()
        if self.train_before_inference:
            self.train()
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem},
        ]
        prompt = self.tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=self.max_seq_length)
        text = self.tokenizer.decode(output[0])
        if not self.inference_mode:
            print(f"Formatted Prompt: {prompt}")
            print(f"Generate Output: {text.split('</think>')[1]}")
        answer = self.extract_answer(text)
        if answer is None:
            answer = 0
        return int(answer)