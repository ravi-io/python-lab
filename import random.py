import random

f = lambda x : -(x - 5) ** 2
x = 0

for _ in range(100):
    step = random.choice([-0.1, 0.1])
    
    if f(x+step) > f(x):
        x += step

print(f"Peak found at x = {round(x,2)}")