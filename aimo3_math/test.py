

import re

a = r"sdsdfsfdsf answer: \boxed{-1, 2}"
b = r"answer: \\boxed\{.*?\}"
match = re.findall(b, a)
print(match)
print(match[-1])

a = "   aa    "
b = r"\s*(\d+)\s*"

res = re.search(b, a)
res2 = re.findall(b, a)
print(res)
# print(type(res.group()))
print(res2)

# r不是针对正则表达式的字面量，而是针对字符串的字面量
# 这里的转义有两层意思
# 加上r之后的转义都是针对正则表达式的转义
pattern = r'^\s*Answer: \\boxed\{\s*(.*?)\s*\}$'
answer = ' Answer: \\boxed{-1}'
res = re.findall(pattern, answer)
print(res)