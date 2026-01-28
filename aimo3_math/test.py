from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/workspace/model/gpt-oss-20b"
gguf_file = "/mnt/workspace/model/gpt-oss-20b-gguf/gpt-oss-20b-Q4_K_M.gguf"

chat_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    gguf_file=gguf_file,
    # load_in_4bit=True,
    device_map="auto"
)

print("model loaded.")