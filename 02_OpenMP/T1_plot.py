import matplotlib.pyplot as plt
import numpy as np

n = np.array([16, 8, 4, 2, 1])
y_option_0 = [12.666682, 23.656514, 46.880126, 96.896767, 204.116529]
y_option_36 = [9.587034, 19.243195, 36.613769, 72.172494, 143.191577]

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