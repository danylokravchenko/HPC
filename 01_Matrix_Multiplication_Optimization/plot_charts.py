import matplotlib.pyplot as plt
import numpy as np

n = np.array([200, 400, 600, 800, 1000])
# T1
T1_y_option_0 = [0.007220, 0.079722, 0.279057, 0.569076, 1.297170]
T1_y_option_1 = [0.005922, 0.045966, 0.156605, 0.361931, 0.725375]
# T1_y_option_1 = [0.002013, 0.014927, 0.049361, 0.116238, 0.229557]

fig, ax = plt.subplots()

opt1_bar = ax.bar(n - 25, T1_y_option_1, width=50, label="Cache aware")
opt0_bar = ax.bar(n + 25, T1_y_option_0, width=50, label="Naive")
ax.bar_label(opt1_bar, fmt="{:0.2}")
ax.bar_label(opt0_bar, fmt="{:0.2}")
ax.set_xticks(n)
ax.legend(loc="upper left", ncols=2)
ax.set_yscale("log")
ax.set_ylabel("Mean execution time")
ax.set_xlabel("Matrix size")

plt.show()

# T3
T3_2D_y = [0.007016, 0.079559, 0.278964, 0.567903, 1.297195]
T3_1D_y = [0.002982, 0.023851, 0.078913, 0.184366, 0.359025]

fig, ax = plt.subplots()

opt1_bar = ax.bar(n - 25, T3_1D_y, width=50, label="1D")
opt0_bar = ax.bar(n + 25, T3_2D_y, width=50, label="2D")
ax.bar_label(opt1_bar, fmt="{:0.2}")
ax.bar_label(opt0_bar, fmt="{:0.2}")
ax.set_xticks(n)
ax.legend(loc="upper left", ncols=2)
ax.set_yscale("log")
ax.set_ylabel("Mean execution time")
ax.set_xlabel("Matrix size")

plt.show()

# T5
T5_y = [0.003327, 0.024428, 0.082108, 0.187177, 0.377977] # cache aware
# T5_y = [0.002396, 0.018860,  0.058958, 0.135571, 0.265216]

fig, ax = plt.subplots()

opt1_bar = ax.bar(n - 25, T5_y, width=50, label="1D+vectorization+IKJ+cacheawareness")
opt0_bar = ax.bar(n + 25, T1_y_option_0, width=50, label="Naive")
ax.bar_label(opt1_bar, fmt="{:0.2}")
ax.bar_label(opt0_bar, fmt="{:0.2}")
ax.set_xticks(n)
ax.legend(loc="upper left", ncols=2)
ax.set_yscale("log")
ax.set_ylabel("Mean execution time")
ax.set_xlabel("Matrix size")

plt.show()