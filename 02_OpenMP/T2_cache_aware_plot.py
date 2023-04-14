import matplotlib.pyplot as plt
import numpy as np

n = np.array([16, 8, 4, 2])
no_schedule = [9.592557, 19.250339, 36.628906, 72.235406]
af = [9.536976, 18.748834, 36.786263, 72.353021]
dynamic = [9.473077, 18.662862, 36.535113, 71.929608]
fac2 = [9.467514, 18.664774, 36.382445, 71.648097]
guided = [9.517252, 18.765597, 36.445530, 71.693121]
static = [9.590226, 19.251630, 36.620387, 72.174246]

fig, ax = plt.subplots()

ax.plot(n, no_schedule, label="No schedule")
ax.plot(n, af, label="af")
ax.plot(n, dynamic, label="DYNAMIC")
ax.plot(n, fac2, label="fac2")
ax.plot(n, guided, label="GUIDED")
ax.plot(n, static, label="STATIC")
ax.set_xticks(n)
ax.set_ylabel("mean execution time")
ax.set_xlabel("Number of threads")
ax.legend(loc="upper right")

plt.show()