import json
from typing import Any, Dict

from datasets import load_dataset, Dataset
from openai import OpenAI
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

from .format import extract_answer, extract_tool_call, exclude_think
from .prompt import system_prompt, preference_prompt


class KaggleSolver:

    def __init__(
            self,
            model_path: str,
            data: Dict[str, Any] = None,
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
        self.dataset = None
        if data is not None:
            self.dataset = load_dataset(**data)
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

    def generate(self, conv):
        if self.offline_mode:
            prompt = self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tools=[]
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_length=self.max_context_length)
            text = self.tokenizer.decode(output[0])
            text = exclude_think(text)
        else:
            response = self.client.chat.completions.create(
                messages=conv,
                model=self.model_path,
                max_tokens=self.max_context_length,
                extra_body={"enable_thinking": self.enable_thinking},
                tools=[tool.get_tool_schema() for tool in self.tools.values()]
            )
            message = response.choices[0].message
            text = message.content
            if message.tool_calls is not None:
                tool_call = message.tool_calls[0]
                text += f'I will use {tool_call.function.name} tool.\n\n<tool_call>{tool_call.model_dump_json()}</tool_call>'

        return text.strip()

    def solve_problem(self, problem: str):
        conv = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Problem: {problem}\n\nPreference: {preference_prompt}"},
        ]
        answer = None
        for t in range(self.max_turns):
            # 无think内容
            response = self.generate(conv)
            print(f"Assistant: \n{response}")
            assis_msg = {"role": "assistant", "content": response}
            conv.append(assis_msg)
            # 不是工具调用，则必然结束
            if "<tool_call>" in response:
                pre_tool_text = response.split("<tool_call>")[0]
                # 提取工具调用，并执行
                tool_call_str = extract_tool_call(response)
                tool_call_json = json.loads(tool_call_str)
                tool_call_msg = {"tool_calls": [tool_call_json]}
                conv[-1]["content"] = pre_tool_text
                conv[-1].update(tool_call_msg)
                # conv.append(tool_call_msg)
                result = self.tool_exec(tool_call_str)
                # 封装消息
                tool_msg = {"role": "tool", "content": result}
                print(f"Tool Exec: \n{result}")
                conv.append(tool_msg)
            elif "\\boxed" in response:
                answer = extract_answer(response)
                break
            else:
                usr_msg = {"role": "user", "content": "continue"}
                conv.append(usr_msg)

        return answer, conv

    # 工具调用：python
    # 多轮对话：会话管理

    def predict(self, problem: str):
        if self.model is None:
            self.load()
        if self.train_before_inference:
            self.train()
            self.train_before_inference = False
        answer, _ = self.solve_problem(problem)
        return answer