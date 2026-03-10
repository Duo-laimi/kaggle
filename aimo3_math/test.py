# from datasets import load_dataset
#
# path = "sample_3500.jsonl"
#
# ds = load_dataset("json", data_files=path)
import re
text = "d   <tool_call>dfsaertetgesrg</tool_call>    "
pattern = r'^\s*<tool_call>(.*?)</tool_call>\s*$'
matches = re.findall(pattern, text, re.DOTALL)
print(matches)