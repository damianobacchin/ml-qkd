import numpy as np
from main import raman_scattering, four_wave_mixing
from config import n_total_channels, n_data_channels, grid_spacing


config = np.zeros(n_total_channels, dtype=int)
config[:n_data_channels] = 1
config[n_data_channels] = 2

results_fwm = []
results_srs = []
for i in range(100):
    np.random.shuffle(config)
    results_fwm.append(four_wave_mixing(config))
    results_srs.append(raman_scattering(config))

print('Mean SRS noise:', np.median(results_srs))
print('Mean FWM noise:', np.median(results_fwm))
