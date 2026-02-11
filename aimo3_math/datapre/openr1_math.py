import os
import re
from openai import OpenAI

from datasets import load_dataset
from functools import partial

DEFAULT_FORMAT_PROMPT = \
r"""
You are an answer formatter. 
User will input an answer to a specific problem. 
The problem will not be inputted. 
Just extract and format the answer and output like 'Answer: \boxed{res1, res2, ...}'.
"""

class QwenFormatter:
    def __init__(self, model, base_url=None, api_key=None, format_prompt: str = None):
        self.model = model
        if base_url is None:
            from dotenv import load_dotenv
            load_dotenv()
            base_url = os.getenv("QWEN_BASE_URL")
            api_key = os.getenv("QWEN_API_KEY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.format_prompt = format_prompt or DEFAULT_FORMAT_PROMPT

    def __call__(self, text: str):
        messages = [
            {"role": "system", "content": self.format_prompt},
            {"role": "user", "content": text}
        ]
        response = self.client.chat.completions.create(messages=messages, model=self.model, stream=False)
        text = response.choices[0].message.content
        pattern = r'Answer: \\boxed\{.*?\}'
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
        return r'Answer: \boxed{-1}'


class FormattedOpenr1Math:
    def __init__(self, path, tokenizer, format_model="qwen-flash", remove_unused=True):
        dataset = load_dataset(path, "default", split="train")
        self.used_columns = ["problem", "solution", "answer"]
        if remove_unused:
            all_columns = dataset.column_names
            unused_columns = []
            for column in all_columns:
                if column not in self.used_columns:
                    unused_columns.append(column)
            dataset = dataset.remove_columns(unused_columns)

        # .filter(lambda example: len(example["solution"]) < 10)
        self.dataset = dataset
        self.formatter = QwenFormatter(format_model)
        self.tokenizer = tokenizer

    def generate_conversation(self, examples):
        problems = examples["problem"]
        solutions = examples["solution"]
        answers = examples["answer"]
        conversations = []
        for problem, solution, answer in zip(problems, solutions, answers):
            formatted = self.formatter(answer)
            conv = [
                {"role": "system", "content": "You are a good math problem solver."},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": f"<think>{solution}</think>\n{formatted}"},
            ]
            conversations.append(conv)
        return {"conversations": conversations}


    def sample(self, n=3500, seed=42, save_to=None, as_prompt=True, batch_size=2):
        def is_normal_answer(example):
            answer = example["conversations"][2]["content"].split("</think>")[1]
            pattern = r'Answer: \\boxed\{\s*(.*?)\s*\}'
            match = re.findall(pattern, answer)
            if len(match) > 0 and match[-1] != "-1":
                return True
            return False

        samples = self.dataset.shuffle(seed).select(range(n)) \
            .map(self.generate_conversation, batched=True, batch_size=batch_size).filter(is_normal_answer)

        if as_prompt:
            samples = samples.map(lambda example: {"text": self.tokenizer.apply_chat_template(example["conversations"], tokenize=False)})

        if save_to is None:
            save_to = f"sample_{n}.jsonl"
        samples.to_json(save_to)
        return samples

