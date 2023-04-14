import matplotlib.pyplot as plt
import numpy as np

n = np.array([16, 8, 4, 2])
no_schedule = [0.592557, 1.250339, 0.628906, 1.235406]
af = [0.536976, 0.748834, 0.786263, 1.353021]
dynamic = [0.473077, 0.662862, 0.535113, 0.929608]
fac2 = [0.467514, 0.664774, 0.382445, 0.648097]
guided = [0.517252, 0.765597, 0.445530, 0.693121]
static = [0.590226, 1.251630, 0.620387, 1.174246]

fig, ax = plt.subplots()

ax.plot(n, no_schedule, label="No schedule")
ax.plot(n, af, label="af")
ax.plot(n, dynamic, label="DYNAMIC")
ax.plot(n, fac2, label="fac2")
ax.plot(n, guided, label="GUIDED")
ax.plot(n, static, label="STATIC")
ax.set_xticks(n)
ax.set_ylabel("difference in execution time")
ax.set_xlabel("Number of threads")
ax.legend(loc="upper right")

plt.show()