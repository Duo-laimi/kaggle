import json
from typing import Any, Dict
from collections import Counter
from threading import Thread

from datasets import load_dataset, Dataset
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from trl import SFTTrainer

from .utils.format import extract_answer, extract_tool_call, exclude_think, extract_think, exclude_tool_call
from .utils.prompt import system_prompt, preference_prompt
from .utils.stream import collect_api_stream, construct_completion_message, collect_model_stream


class KaggleSolver:

    def __init__(
            self,
            model_path: str,
            data: Dict[str, Any] = None,
            base_url: str = None,
            api_key: str = None,
            max_turns: int = 128,
            max_tries: int = 1,
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
        self.dataset = None
        if data is not None:
            self.dataset = load_dataset(**data)
        self.offline_mode = True
        if base_url is not None:
            self.offline_mode = False
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_turns = max_turns
        self.max_tries = max_tries
        self.enable_thinking = enable_thinking
        self.max_context_length = max_context_length
        self.train_before_inference = train_before_inference and self.offline_mode
        self.use_peft_train = use_peft_train
        self.peft_weight_save_path = peft_weight_save_path
        self.quantize_base_model = quantize_base_model
        self.peft_config = peft_config
        self.sft_config = sft_config
        self.quantization_config = quantization_config
        self.model_kwargs = kwargs
        self.system_prompt = f"{system_prompt}"
        self.tools = None

    def load(self):
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self.quantization_config,
            **self.model_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # 2. 设置 Streamer
        self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.model.config.use_cache = True
        # self.collator =
        print(f"Successfully load model from {self.model_path}.")

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset

    def bind_tools(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def tool_exec(self, tool_call_str):
        tool_call = json.loads(tool_call_str)["function"]
        name = tool_call["name"]
        kwargs = tool_call["arguments"]
        kwargs = json.loads(kwargs)
        if name not in self.tools:
            return f"ERROR: Tool {name} not found."
        result = self.tools[name](**kwargs)
        return str(result)

    def train(self):
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

    def reply_iter(self, conv):
        if self.offline_mode:
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tools=[]
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            # output = self.model.generate(**inputs, max_length=self.max_context_length)
            kwargs = dict(**inputs, streamer=self.streamer, max_length=self.max_context_length)
            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            content, reasoning_content, tool_calls = collect_model_stream(self.streamer)
            message = construct_completion_message(content, reasoning_content, tool_calls)
        else:
            response = self.client.chat.completions.create(
                messages=conv,
                model=self.model_path,
                max_tokens=self.max_context_length,
                extra_body={"enable_thinking": self.enable_thinking},
                tools=[tool.get_tool_schema() for tool in self.tools.values()],
                stream=True
            )
            # message = response.choices[0].message
            content, reasoning_content, tool_calls = collect_api_stream(response)

            message = construct_completion_message(content, reasoning_content, tool_calls)

        return message

    def solve_problem(self, problem: str):
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem}\n\nPreference: {preference_prompt}"},
        ]
        answer = None
        for t in range(self.max_turns):
            # 无think内容
            message = self.reply_iter(conv)
            # print(f"Assistant: \n{message}")
            conv.append(message)
            # 不是工具调用，则必然结束
            if message.tool_calls is not None:
                # 提取工具调用，并执行
                tool_call = message.tool_calls[0]
                # conv.append(tool_call_msg)
                result = self.tool_exec(tool_call.model_dump_json())
                # 封装消息
                tool_msg = {"role": "tool", "content": result}
                print(f"Tool Exec: \n{result}")
                conv.append(tool_msg)
            elif "\\boxed" in message.content:
                answer = extract_answer(message.content)
                break
            else:
                usr_msg = {"role": "user", "content": "continue"}
                conv.append(usr_msg)

        return answer, conv

    def multiple_check_answer(self, problem: str):
        answers = []
        for _ in range(self.max_tries):
            answer, _ = self.solve_problem(problem)
            answers.append(answer)
        counter = Counter(answers)
        if 0 in counter:
            counter.pop(0)
        if len(counter) == 0:
            return 0
        return counter.most_common(1)[0][0]

    # 工具调用：python
    # 多轮对话：会话管理

    def predict(self, problem: str):
        if self.model is None and self.offline_mode:
            self.load()
        if self.train_before_inference:
            self.train()
            self.train_before_inference = False
        answer = self.multiple_check_answer(problem)
        return answer