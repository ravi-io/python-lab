import random

f = lambda x: -((x - 3) ** 2) + 10
curr = random.uniform(-10, 10)

for _ in range(1000):
    nxt = curr + random.choice([-0.1, 0.1])
    if f(nxt) > f(curr):
        curr = nxt

print(f"Max at x = {curr:.2f}, f(x) = {f(curr):.2f}")