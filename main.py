from math import exp, pi, sin
import numpy as np
from matplotlib import pyplot as plt
from statistics import mean
from raman import raman_cross_section as raman_cross_section_tab
from random import random, randint
from config import n_data_channels, n_total_channels, start_frequency, grid_spacing, c0, quantum_receiver_bandwidth, fiber_length, fiber_attenuation, power_input,\
    fiber_dispersion, fiber_dispersion_slope, nonlinear_refractive_index, effective_fiber_cross_section_area


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

    att_param = exp(-fiber_attenuation*fiber_length) #pow(10, (-fiber_attenuation*fiber_length)/10)

    for i in data_channels:
        ch_i_frequency = c0 / grid_wavelength(i)
        for j in data_channels:
            ch_j_frequency = c0 / grid_wavelength(j)
            for k in data_channels:
                ch_k_frequency = c0 / grid_wavelength(k)
                if ch_i_frequency + ch_j_frequency - ch_k_frequency == quantum_frequency:
                    quantum_wavelength = grid_wavelength(quantum_channel)
                    phase_matching_factor = ( 2*pi*pow(quantum_wavelength, 2)/c0 ) * abs(ch_i_frequency-ch_k_frequency) * abs(ch_j_frequency-ch_k_frequency) * ( fiber_dispersion + fiber_dispersion_slope * pow(quantum_wavelength, 2)/c0 * ( abs(ch_i_frequency-ch_k_frequency) + abs(ch_j_frequency-ch_k_frequency) ) )
                    phase_matching_efficiency = fiber_attenuation**2/(fiber_attenuation**2 + phase_matching_factor**2) * ( 1 + 4*att_param * ( pow(sin(phase_matching_factor * fiber_length / 2), 2) / (1 - att_param)**2 ) )
                    if i==j: degeneracy_param = 1
                    else: degeneracy_param = 2
                    power_output = phase_matching_efficiency * third_order_nonlinear_coefficient**2 * power_input**3 * degeneracy_param**2 * att_param * ( (1 - att_param)**2 / (9 * fiber_attenuation**2) )
                    fwm_noise += power_output
    return fwm_noise

def fitness_function(config):
    rs = raman_scattering(config)
    fwm = four_wave_mixing(config)
    return rs + fwm


def simulated_annealing(init_config, Tmax=60, Tmin=1, alpha=0.99, mod=False):
    current_config = init_config.copy()

    count_x = 1
    x_ax = []
    y_ax = []

    t = Tmax

    while t > Tmin:
        if mod:
            idx_a = np.random.choice(np.where(current_config != 2)[0])
            idx_b = np.random.choice(np.where((current_config != 2) & (current_config != current_config[idx_a]))[0])
        else:
            idx_a = np.random.choice(len(current_config))
            idx_b = np.random.choice(np.where(current_config != current_config[idx_a])[0])

        candidate_config = current_config.copy()
        candidate_config[idx_a] = current_config[idx_b]
        candidate_config[idx_b] = current_config[idx_a]

        fitness_current = fitness_function(current_config)
        fitness_candidate = fitness_function(candidate_config)
        
        delta = fitness_candidate - fitness_current

        if delta < 0:
            current_config = candidate_config
        else:
            delta_prop = (delta / fitness_candidate) * 200
            if random() < exp(-delta_prop/t):
                current_config = candidate_config

        t *= alpha

        x_ax.append(count_x)
        count_x += 1
        y_ax.append(fitness_function(current_config))

    # plt.figure(figsize=(10, 6))
    # plt.plot(x_ax, y_ax)
    # plt.grid(True, linestyle = '--', linewidth = 0.5)
    # plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
    # plt.show()

    return current_config

# Create config
# config = np.zeros(n_total_channels, dtype=int)
# config[:n_data_channels] = 1
# config[n_data_channels] = 2

# config = np.array([2, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
# print(raman_scattering(config))
# print(four_wave_mixing(config))
# print(fitness_function(config))


#np.random.shuffle(config)

# results_stat = np.zeros(n_total_channels, dtype=float)
# results = np.zeros(n_total_channels, dtype=float)

# for i in range(1, n_total_channels):
#     print('>>> Ottimizzazione', i, 'canali')

#     # config = np.zeros(n_total_channels, dtype=int)
#     # config[:i] = 1
#     # config[i] = 2

#     # Random configs
#     arr = np.zeros((50), dtype=float)
#     for j in range(50):
#         config_perm = np.zeros(n_total_channels-1, dtype=int)
#         config_perm[:i] = 1
#         np.random.shuffle(config_perm)
#         config = np.concatenate(([2], config_perm))
#         arr[j] = fitness_function(config)
#     results_stat[i] = arr.mean()

#     # Annealing
#     results[i] = fitness_function(simulated_annealing(config))

# np.savetxt('results.txt', results)
# np.savetxt('results_stat.txt', results_stat)


results_stat = np.loadtxt('results_stat.txt')
results = np.loadtxt('results.txt')

# plt.figure(figsize=(6, 4))

# plt.plot(np.arange(1, n_total_channels), results_stat[1:], '-', label='Average of 50 FBCA')
# plt.plot(np.arange(1, n_total_channels), results[1:], '-', label='Simulated annealing')

# plt.scatter(np.arange(1, n_total_channels), results_stat[1:], marker='o')
# plt.scatter(np.arange(1, n_total_channels), results[1:], marker='o')

# plt.legend()
# plt.grid(True, linestyle = '--', linewidth = 0.5)
# plt.xlabel('Number of data channels')
# plt.ylabel('Noise power [W]')

# plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
# plt.show()



# OPT PERCENTAGE

opt_perc = np.zeros(n_total_channels, dtype=float)
for i in range(len(results_stat)):
    opt_perc[i] = (results_stat[i] - results[i]) / results_stat[i] *100

plt.figure(figsize=(6, 3))

plt.plot(np.arange(1, n_total_channels), opt_perc[1:], '-')
plt.scatter(np.arange(1, n_total_channels), opt_perc[1:], marker='o')
plt.grid(True, linestyle = '--', linewidth = 0.5)
plt.xlabel('Number of data channels')
plt.ylabel('Percentage of optimization [%]')

plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
plt.show()




# x_ax = []
# srs = []
# fwm = []
# for grid in [100e9, 50e9, 25e9, 12.5e9]:
#     grid_spacing = grid
#     config = np.zeros(n_total_channels, dtype=int)
#     config[:n_data_channels] = 1
#     config[n_data_channels] = 2

#     srs_p = np.zeros(10, dtype=float)
#     fwm_p = np.zeros(10, dtype=float)
#     for i in range(10):
#         np.random.shuffle(config)
#         srs_p[i] = raman_scattering(config)
#         fwm_p[i] = four_wave_mixing(config)
#     srs.append(srs_p.mean())
#     fwm.append(fwm_p.mean())
#     x_ax.append(grid)

# plt.figure(figsize=(6, 3))
# plt.grid(True, linestyle = '--', linewidth = 0.5)
# plt.plot(x_ax, srs, '-')
# plt.plot(x_ax, fwm, '-')
# plt.scatter(x_ax, srs, label='SRS')
# plt.scatter(x_ax, fwm, label='FWM')
# plt.xlabel('Grid spacing [Hz]')
# plt.ylabel('Noise power [W]')
# plt.legend()
# plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
# plt.show()
