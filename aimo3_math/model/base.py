import torch
import torch.nn as nn
from unsloth import FastLanguageModel


class KaggleSolver:

    def __init__(
            self,
            model_name,
            dtype=None,
            max_seq_length=1024,
            inference_mode: bool = True
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.inference_mode = inference_mode
        if not inference_mode:
            self.load()

    def load(self):
        """Simulate model loading."""
        print("Loading model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            dtype=self.dtype,  # None for auto detection
            max_seq_length=self.max_seq_length,  # Choose any for long context!
            load_in_4bit=True,  # 4 bit quantization to reduce memory
            full_finetuning=False,  # [NEW!] We have full finetuning now!
        )
        if not self.inference_mode:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj", ],
                lora_alpha=16,
                lora_dropout=0,  # Supports any, but = 0 is optimized
                bias="none",  # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
                random_state=3407,
                use_rslora=False,  # We support rank stabilized LoRA
                loftq_config=None,  # And LoftQ
            )
        return self.model

    @torch.no_grad()
    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self.model is None:
            self.model = self.load()
            
        
        return self.model(problem)
