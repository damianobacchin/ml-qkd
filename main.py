from math import exp
import numpy as np
from matplotlib import pyplot as plt
from raman import raman_cross_section as raman_cross_section_tab

from config import config_1, start_frequency, grid_spacing, c0

quantum_receiver_bandwidth = 0.6e-9 # (0.6-0.8)nm
fiber_length = 50 # (1-100)km
fiber_attenuation = 0.2 # (0.16-0.185-0.195-0.21-0.3)dB/km
power_input = 1

# plt.plot(raman_cross_section.keys(), raman_cross_section.values())
# plt.show()

def grid_wavelength(n):
    frequency = start_frequency + n * grid_spacing
    return c0 / frequency

def raman_cross_section(data_wavelength, quantum_wavelength):
    lambda_delta = ( (1/1550e-9) + (1/data_wavelength) - (1/quantum_wavelength) )**-1
    result = (lambda_delta/quantum_wavelength)**4 * raman_cross_section_tab[round(lambda_delta*10**9)]
    return result

def raman_scattering(config):
    data_channels = np.where(config == 1)[0]
    quantum_channel = np.where(config == 2)[0][0]
    _sum = 0
    for channel in data_channels:
        _sum += raman_cross_section(grid_wavelength(channel), grid_wavelength(quantum_channel))

    return power_input * exp(-fiber_attenuation*fiber_length) + _sum

def fitness_function(config):
    pass


noise = raman_scattering(config_1)
print(noise)