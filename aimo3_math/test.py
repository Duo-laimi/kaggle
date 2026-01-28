from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/workspace/model/gpt-oss-20b"

chat_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # load_in_4bit=True,
    device_map="auto"
)

print("model loaded.")