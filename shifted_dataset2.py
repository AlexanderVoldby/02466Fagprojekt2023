import numpy as np
import matplotlib.pyplot as plt
import torch
from ShiftNMF import ShiftNMF
np.random.seed(42069)
# Define random sources, mixings and shifts; H, W and tau
N, M, d = 10, 10000, 3
Fs = 1000  # The sampling frequency we use for the simulation
t0 = 10     # The half-time interval we look at
t = np.arange(-t0, t0, 1/Fs)  # the time samples
f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

def ft(samples, Fs, t0):
    """Approximate the Fourier Transform of a time-limited
    signal by means of the discrete Fourier Transform.

    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal. We assume it is zero.
    """
    f = np.linspace(-Fs / 2, Fs / 2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples) / Fs * np.exp(-2j * np.pi * f * t0))

def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)


# Random mixings:
W = np.random.rand(N, d)
# Random gaussian shifts
tau = np.random.randn(N, d)*200
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 7]
t = np.arange(0, 1000, 0.1)
H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

# Plot original signals
plt.figure()
for signal in H:
    plt.plot(signal)
plt.title("3 original sources")
plt.show()

plt.figure()
plt.plot(H[0])
plt.title("Initial signal")
plt.show()

shift = 200.435
shifted_signal = np.fft.fft(H[0])*np.exp(-2j*np.pi*np.arange(0, M)/M*shift)
plt.figure()
plt.plot(np.fft.ifft(shifted_signal), label="Shifted")
plt.plot(H[0], label="Original signal")
plt.legend()
plt.title("Signal after applying shift of 20")
plt.show()

# Create artifically shifted dataset
# Fourier transform of H
Hf = np.fft.fft(H)
# The matrix that approximates the observations
WHt = np.empty((N, M))
f = np.arange(0, M)


for i in range(N):
    signal = np.zeros(M, dtype=complex)
    for j in range(d):
        signal += W[i, j] * np.fft.ifft(Hf[j]*np.exp(-2j * np.pi * f / M * tau[i, j]))
    WHt[i] = signal.real

# Plot the shifted observations in WHt
plt.figure()
for signal in WHt:
    plt.plot(signal)
plt.title("10 signals made by mixing shifts of the original 3 signals")
plt.show()