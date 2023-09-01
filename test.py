import numpy as np
from matplotlib import pyplot as plt

q_ch = {
    1: 0.04,
    2: 0.34,
    3: 0.57,
    4: 0.17,
    5: 0.00,
    6: 0.00,
    7: 0.00,
    8: 0.89,
    9: 0.99,
    10: 0.75,
    11: 0.08,
    12: 0.01
}

keys = list(q_ch.keys())
values = list(q_ch.values())

plt.figure(figsize=(7, 3))
plt.bar(keys, values)
plt.grid(True, linestyle = '--', linewidth = 0.5)
plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
plt.show()