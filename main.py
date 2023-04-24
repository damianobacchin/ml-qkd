import numpy as np
from matplotlib import pyplot as plt
from raman import raman_cross_section

plt.plot(raman_cross_section.keys(), raman_cross_section.values())
plt.show()