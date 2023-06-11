c0 = 299792458

n_data_channels = 5
n_total_channels = 8

start_frequency = 191.6e12
grid_spacing = 100e9

quantum_receiver_bandwidth = 0.6e-9 #(0.6-0.8)nm
fiber_length = 50e3 #(1-100)km
fiber_attenuation = 0.2e-3 #(0.16, 0.185, 0.195, 0.21, 0.3)dB/km
power_input = 10e-3 #(1-10)dBm

nonlinear_refractive_index = 2.6e-20 # (2.6, 2.8, 3.2) m^2/W
fiber_dispersion = 0.16e-6 # (0.1, 0.16, 4.25, 20.35) ps/nm/km
fiber_dispersion_slope = 0.06e3 # ps/nm^2/km
effective_fiber_cross_section_area = 50e-12 # (80, 50) um^2
p = 1 # m