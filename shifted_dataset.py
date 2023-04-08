import numpy as np
import matplotlib.pyplot as plt
import torch
from ShiftNMF import ShiftNMF

# Define random sources, mixings and shifts; H, W and tau
N, M, d = 10, 10000, 3
Fs = 10000  # The sampling frequency we use for the simulation
t0 = 10     # The half-time interval we look at
t = np.arange(-t0, t0, 1/Fs)  # the time samples
f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples
g = lambda t, f: np.sin(2*np.pi*f*2*t)
# Create 10 sin waves

def ft(samples, Fs, t0):
    """Approximate the Fourier Transform of a time-limited
    signal by means of the discrete Fourier Transform.

    samples: signal values sampled at the positions t0 + n/Fs
    Fs: Sampling frequency of the signal
    t0: starting time of the sampling of the signal
    """
    f = np.linspace(-Fs / 2, Fs / 2, len(samples), endpoint=False)
    return np.fft.fftshift(np.fft.fft(samples) / Fs * np.exp(-2j * np.pi * f * t0))

# sources = np.array([np.sin(freqs[i]*x) for i in range(d)])
# sources.reshape((d, M))
plt.figure()
plt.plot(t, g(t, 1))
plt.show()

plt.figure()
plt.plot(f, ft(g(t, 1), Fs, t0).real)
plt.show()

plt.figure()
plt.plot(f, np.fft.fft(g(t, 1)).real)
plt.show()
# Random mixings:
W = np.random.rand(N, d)
# Random gaussian shifts
t = np.random.randn(N, d)*5
# Rebuild data matrix. I use eq. 7 from the ShiftNMF paper
X = np.empty((N, M))
for n in range(N):
    # Define delayed version of the source to the n'th sensor
    omega = np.array([np.array([np.exp(-2*np.pi*1j*f/M) for f in range(M)]) * t[n, col] for col in range(d)])
    H_delay = np.fft.ifft(np.fft.fft(H)*omega)
    X[n] = np.matmul(W[n], H_delay)


plt.figure()
#for source in sources:
plt.plot(x, sources[0])
plt.title("Original signals")
plt.show()

plt.figure()
for source in H:
    plt.plot(source)
plt.title("3 Original sources")
plt.show()


