

import re

a = r"sdsdfsfdsf answer: \boxed{-1, 2}"
b = r"answer: \\boxed\{.*?\}"
match = re.findall(b, a)
print(match)
print(match[-1])

a = "123"
b = r"^\d+$"
if re.match(b, a):
    print(a)
