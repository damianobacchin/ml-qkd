import numpy as np

c0 = 299792458

n_data_channels = 5
n_total_channels = 10

start_frequency = 191.6e12
grid_spacing = 100e9

quantum_receiver_bandwidth = 0.6e-9 # (0.6-0.8)nm
fiber_length = 50 # (1-100)km
fiber_attenuation = 0.2 # (0.16-0.185-0.195-0.21-0.3)dB/km
power_input = 1