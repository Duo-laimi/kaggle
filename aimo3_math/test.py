from datasets import load_dataset

path = "sample_3500.jsonl"

ds = load_dataset("json", data_files=path)