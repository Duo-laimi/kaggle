import re

def extract_think(text):
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, text, re.DOTALL)
    if len(matches) == 0:
        return None
    return matches[0].strip()

def exclude_think(text):
    if "</think>" not in text:
        return text
    splits = text.split("</think>")
    return splits[-1]

def exclude_tool_call(text):
    if "<tool_call>" not in text:
        return text
    splits = text.split("<tool_call>")
    return splits[0]

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
        try:
            ans = matches[-1].strip()
            return int(ans)
        except Exception:
            pass
    return 0