import matplotlib.pyplot as plt
import numpy as np

n = np.array([16, 8, 4, 2])
y_option_0 = [12.719382, 23.663764, 46.898695, 96.937390]
y_option_36 = [9.592155, 19.237361, 36.623906, 72.176554]

fig, ax = plt.subplots()

opt1_bar = ax.bar(n - 0.125, y_option_36, width=0.25, label="Cache aware")
opt0_bar = ax.bar(n + 0.125, y_option_0, width=0.25, label="Naive", color="#90ee90")
ax.bar_label(opt1_bar, fmt="{:0.4}")
ax.bar_label(opt0_bar, fmt="{:0.4}")
ax.set_xticks(n)
ax.legend(loc="upper right", ncols=2)
ax.set_yscale("log")
ax.set_ylabel("log mean execution time")
ax.set_xlabel("Number of threads")

plt.show()