import re


def exclude_think(text):
    if "</think>" not in text:
        return text
    splits = text.split("</think>")
    return splits[-1]


def extract_tool_call(text):
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return None
    return matches[-1].strip()

def extract_answer(text):
    text = exclude_think(text)
    # 2. 匹配 \boxed{内容} 中的内容
    # 使用捕获组 () 来提取数字，并处理可能存在的空格
    pattern = r"\\boxed{([^{}]+)}"
    matches = re.findall(pattern, text)
    # 3. 返回最后一个匹配项
    if matches:
        return matches[-1].strip()
    return None