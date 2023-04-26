from math import exp
import numpy as np
from matplotlib import pyplot as plt
from raman import raman_cross_section as raman_cross_section_tab
from random import randint

from config import n_data_channels, n_total_channels, start_frequency, grid_spacing, c0, quantum_receiver_bandwidth, fiber_length, fiber_attenuation, power_input

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
        _sum += raman_cross_section(grid_wavelength(channel), grid_wavelength(quantum_channel)) * quantum_receiver_bandwidth

    return power_input * exp(-fiber_attenuation*fiber_length) + _sum

def fitness_function(config):
    rs = raman_scattering(config)
    # TODO four wave mixing
    # TODO adjacent channel crosstalk
    return rs



def simulated_annealing(init_config):
    config = init_config.copy()

    for t in np.linspace(100, 1, 10000):
        idx_a = randint(0, n_total_channels-1)
        idx_b = randint(0, n_total_channels-1)

        new_config = config.copy()
        new_config[idx_a] = config[idx_b]
        new_config[idx_b] = config[idx_a]

        fitness_init = fitness_function(config)
        fitness_new = fitness_function(new_config)

        delta = fitness_new - fitness_init

        if delta < 0:
            config = new_config
        else:
            rand = randint(0, 100)
            if rand < exp(delta/t):
                config = new_config

    return config

# Create config
config = np.zeros(n_total_channels, dtype=int)
config[:n_data_channels] = 1
config[n_data_channels] = 2

result = simulated_annealing(config)
print(result)