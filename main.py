import modFreq as modf
import digital_rf as drf
import matplotlib.pyplot as plt
import numpy as np

# Files location .hdf5
do = drf.DigitalRFReader('/home/david/Documents/DATA/CHIRP@2025-10-07T19-57-06/rawdata')

# Rawdata properties
print(do.get_properties('ch0'))

# Chirp parameters
A = 1.0
factor = 1200.0
A_echo = A * factor
# PRI (s)
ipp = 400.0e-6          
dc = 12.0
sr_tx = 20.0e6
sr_rx = 2.5e6
fc = 0.0e6
bw = 1.0e6
td_ = 200
window_ = 'B'
mode_f_ = 0
phi_ = 0

# Doppler parameters
velocity = 10.0         # m/s
freq_op_ = 9.345e9      # Hz (frecuencia de operación)
c_ = 3.0e8

# Doppler frequency
f_d = 2.0 * velocity * freq_op_ / c_

# Phase increment per profile
phi_factor = 2.0 * np.pi * f_d * ipp

print(f"Doppler frequency: {f_d:.2f} Hz")
print(f"Phase increment per profile: {phi_factor:.4f} rad")

# Chirp generation
chirp, full_chirp = modf.chirpMod(A_echo,
                                  ipp,
                                  dc,
                                  sr_rx,
                                  sr_rx,
                                  fc,
                                  bw,
                                  t_d=td_,
                                  window=window_,
                                  mode_f=mode_f_,
                                  phi=phi_)

# Number of samples and profiles
number_samples = int(round(ipp * sr_rx))
number_profiles = 500

# Digital RF bounds
first, last = do.get_bounds('ch0')
s0 = first

# Read first 500 profiles
for p in range(number_profiles):

    s = s0 + p * number_samples

    if s + number_samples > last:
        print("No hay más datos disponibles.")
        break

    # Read raw profile
    data = do.read_vector(s,
                          number_samples,
                          'ch0')

    # Doppler ONLY on the chirp
    phase_p = np.exp(1j * p * phi_factor)
    chirp_doppler = full_chirp * phase_p

    # Superposition: raw data + Doppler-shifted chirp
    data_ = data + chirp_doppler

    # Plot
    time_ = np.arange(len(data_)) / sr_rx

    plt.figure(figsize=(10, 4))
    plt.plot(time_, np.real(data_), label="Real")
    plt.plot(time_, np.imag(data_), label="Imag", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Profile {p}")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Perfil {p}, shape: {data_.shape}")
