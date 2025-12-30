# ==========================================
# Earthquake Signal Analysis using FFT
# Applied Maths Capstone Project
# ==========================================

from obspy import read
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# -------------------------------
# PATHS (UPDATE IF NEEDED)
# -------------------------------
DATA_PATH = "data/123319.mseed"     # <-- update only if filename differs
RESULTS_DIR = "results"

# Create results folder if not exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------
# STEP 1: READ MINI-SEED FILE
# -------------------------------
stream = read(DATA_PATH)

# Select vertical (Z) component
z_trace = stream.select(component="Z")[0]

signal = z_trace.data.astype(float)
sampling_rate = z_trace.stats.sampling_rate
n_samples = len(signal)

time = np.arange(n_samples) / sampling_rate

# -------------------------------
# STEP 2: RAW SIGNAL PLOT
# -------------------------------
plt.figure(figsize=(10, 4))
plt.plot(time, signal, color="black")
plt.title("Raw Seismic Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{RESULTS_DIR}/time_domain_raw.png", dpi=300)
plt.close()   # IMPORTANT (prevents empty images)

# -------------------------------
# STEP 3: PRE-PROCESSING
# -------------------------------
# Mean removal
signal_mean_removed = signal - np.mean(signal)

# Normalization
signal_normalized = signal_mean_removed / np.max(np.abs(signal_mean_removed))

# Plot pre-processed signal
plt.figure(figsize=(10, 4))
plt.plot(time, signal_normalized, color="blue")
plt.title("Pre-processed Seismic Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Normalized Amplitude")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{RESULTS_DIR}/time_domain_preprocessed.png", dpi=300)
plt.close()

# -------------------------------
# STEP 4: FFT ANALYSIS
# -------------------------------
fft_values = fft(signal_normalized)
frequencies = fftfreq(n_samples, d=1/sampling_rate)

# Use only positive frequencies
positive_freqs = frequencies[:n_samples // 2]
magnitude = np.abs(fft_values[:n_samples // 2])

# Plot FFT spectrum
plt.figure(figsize=(10, 4))
plt.plot(positive_freqs, magnitude, color="red")
plt.title("FFT Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"{RESULTS_DIR}/fft_spectrum.png", dpi=300)
plt.close()

# -------------------------------
# STEP 5: DOMINANT FREQUENCY
# -------------------------------
dominant_frequency = positive_freqs[np.argmax(magnitude)]

# -------------------------------
# STEP 6: DAMAGE RISK ASSESSMENT
# -------------------------------
if dominant_frequency < 1:
    damage_risk = "Low"
elif dominant_frequency < 5:
    damage_risk = "Moderate"
else:
    damage_risk = "High"

# -------------------------------
# FINAL OUTPUT
# -------------------------------
print("========== ANALYSIS RESULTS ==========")
print(f"Sampling Rate        : {sampling_rate} Hz")
print(f"Dominant Frequency   : {dominant_frequency:.2f} Hz")
print(f"Estimated Damage Risk: {damage_risk}")
print("Plots saved in 'results/' folder")
