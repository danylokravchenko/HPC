import matplotlib.pyplot as plt
import numpy as np

n = np.array([16, 8, 4, 2])
no_schedule = [12.631609, 23.654623, 46.875944, 96.881314]
af = [12.661289, 23.822520, 47.279767, 98.278620]
dynamic = [35.133957, 63.040021, 81.725453, 127.20787]
fac2 = [12.479275, 23.862324, 47.380583, 97.218500]
guided = [12.450985, 23.780721, 47.110249, 98.290247]
static = [12.591200, 23.659383, 46.899315, 96.992544]

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