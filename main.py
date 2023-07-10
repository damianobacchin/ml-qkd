from math import exp, pi, sin
import numpy as np
from matplotlib import pyplot as plt
from raman import raman_cross_section as raman_cross_section_tab
from random import randint
from config import n_data_channels, n_total_channels, start_frequency, grid_spacing, c0, quantum_receiver_bandwidth, fiber_length, fiber_attenuation, power_input,\
    fiber_dispersion, fiber_dispersion_slope, nonlinear_refractive_index, effective_fiber_cross_section_area, p


def grid_wavelength(n):
    frequency = start_frequency + n * grid_spacing
    return c0 / frequency

def raman_cross_section(data_wavelength, quantum_wavelength):
    lambda_delta = ( (1/1550e-9) + (1/data_wavelength) - (1/quantum_wavelength) )**-1
    result = pow(lambda_delta/quantum_wavelength, 4) * raman_cross_section_tab[round(lambda_delta*10**9)]
    return result

def raman_scattering(config):
    data_channels = np.where(config == 1)[0]
    quantum_channel = np.where(config == 2)[0][0]
    _sum = 0
    for channel in data_channels:
        _sum += raman_cross_section(grid_wavelength(channel), grid_wavelength(quantum_channel)) * quantum_receiver_bandwidth

    return power_input * pow(10, (-fiber_attenuation*fiber_length)/10) * fiber_length * _sum


def four_wave_mixing(config):
    data_channels = np.where(config == 1)[0]
    quantum_channel = np.where(config == 2)[0][0]

    quantum_frequency = c0 / grid_wavelength(quantum_channel)
    fwm_noise = 0
    third_order_nonlinear_coefficient = 2*pi*nonlinear_refractive_index/(1550e-9*effective_fiber_cross_section_area)
    for i in data_channels:
        ch_i_frequency = c0 / grid_wavelength(i)
        for j in data_channels:
            ch_j_frequency = c0 / grid_wavelength(j)
            for k in data_channels:
                ch_k_frequency = c0 / grid_wavelength(k)
                if ch_i_frequency + ch_j_frequency - ch_k_frequency == quantum_frequency:
                    quantum_wavelength = grid_wavelength(quantum_channel)
                    phase_matching_factor = ( 2*pi*pow(quantum_wavelength, 2)/c0 ) * abs(ch_i_frequency-ch_k_frequency) * abs(ch_j_frequency-ch_k_frequency) * ( fiber_dispersion + fiber_dispersion_slope * pow(quantum_wavelength, 2)/c0 * ( abs(ch_i_frequency-ch_k_frequency) + abs(ch_j_frequency-ch_k_frequency) ) )
                    phase_matching_efficiency = fiber_attenuation**2/(fiber_attenuation**2 + phase_matching_factor**2) * ( 1 + 4*pow(10, (-fiber_attenuation*fiber_length)/10) * ( pow(sin(phase_matching_factor * fiber_length / 2), 2) / (1 - pow(10, (-fiber_attenuation*fiber_length)/10))**2 ) )
                    if i==j: degeneracy_param = 1
                    else: degeneracy_param = 2
                    power_output = phase_matching_efficiency * third_order_nonlinear_coefficient**2 * p**2 * power_input**3 * degeneracy_param**2 * pow(10, (-fiber_attenuation*fiber_length)/10) * ( (1 - pow(10, (-fiber_attenuation*fiber_length)/10))**2 / (9 * fiber_attenuation**2) )
                    fwm_noise += power_output
    return fwm_noise

def fitness_function(config):
    rs = raman_scattering(config)
    #fwm = four_wave_mixing(config)
    return rs# + fwm


def simulated_annealing(init_config):
    config = init_config.copy()
    min_config = config.copy()

    for t in np.linspace(100, 1, 5000):
        idx_a = randint(0, n_total_channels-1)
        idx_b = randint(0, n_total_channels-1)

        new_config = config.copy()
        new_config[idx_a] = config[idx_b]
        new_config[idx_b] = config[idx_a]

        fitness_init = fitness_function(config)
        fitness_new = fitness_function(new_config)

        if fitness_new < fitness_function(min_config):
            min_config = new_config

        delta = fitness_new - fitness_init

        if delta < 0:
            config = new_config
        else:
            rand = randint(0, 100)
            if rand < exp(delta/t):
                config = new_config

    if fitness_function(min_config) < fitness_function(config):
        return min_config
    return config

# Create config
config = np.zeros(n_total_channels, dtype=int)
config[:n_data_channels] = 1
config[n_data_channels] = 2

#print('Raman scattering:', raman_scattering(config))
#print('Four wave mixing:', four_wave_mixing(config))

# result = simulated_annealing(config)
# print(result, fitness_function(result))

# spacings = [200, 100, 50, 25, 12.5]
# y1 = []
# y2 = []
# for spacing in spacings:
#     grid_spacing = spacing * 1e9
#     results_srs = raman_scattering(config)
#     results_fwm = four_wave_mixing(config)
    
#     y1.append(np.array(results_srs))
#     y2.append(np.array(results_fwm))

# plt.plot(np.array(spacings), y1, label='SRS')
# plt.plot(np.array(spacings), y2, label='FWM')

# plt.xlabel('Grid spacing (GHz)')
# plt.ylabel('Noise (W)')
# plt.title('Total noise vs grid spacing')
# plt.grid(linewidth=0.5)
# plt.legend()

# Save image
#plt.show()
#plt.savefig('total.png', dpi=300, bbox_inches='tight')

