import digital_rf as drf
import matplotlib.pyplot as plt
import numpy as np
import time

# Files location .hdf5
do = drf.DigitalRFReader('/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata')

# Rawdata properties
print(do.get_properties('ch0'))

# Number of samples
IPP = 400*1e-6
SR_RX = 2.5*1e6
number_samples = int(round(IPP*SR_RX))

number_profiles = 500

first, last = do.get_bounds('ch0')
s0 = first

# Read first 500 profiles
for p in range(number_profiles):
    s = s0 + p * number_samples

    if s + number_samples > last:
        break

    data = do.read_vector(s, number_samples, 'ch0')
    time_ = np.arange(len(data))
    plt.plot(time_, np.real(data))
    plt.plot(time_, np.imag(data))
    plt.show()
    print(f"Perfil {p}, shape:", data.shape)
