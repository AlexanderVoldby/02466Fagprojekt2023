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
tau = np.random.randn(N, d)*50
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 7]
t = np.arange(0, 1000, 0.1)
H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])
plt.figure()
for signal in H:
    plt.plot(ft(signal, Fs, t0))
plt.show()

# Rebuild data matrix. I use eq. 7 from the ShiftNMF paper
X = [None]*N
f = np.arange(0, 10000)
for n in range(N):
    # Define delayed version of the d'th source to the n'th sensor
    omega = np.array([np.exp(-2j*np.pi*f/M * tau[n, col]) for col in range(d)])
    H_delay = np.fft.ifft(np.fft.fft(H)*omega)
    if n == 0:
        plt.figure()
        for signal in H_delay:
            plt.plot(signal)
        plt.title("Delayed version of the source signal(s) to the 0'th channel")
        plt.show()
    X[n] = np.matmul(W[n], H_delay)

# Plot original signals
plt.figure()
for signal in H:
    plt.plot(signal)
plt.title("3 original sources")
plt.show()

plt.figure()
for signal in X:
    plt.plot(signal.real)
plt.title("Dataset build from mixing and shifts of the three sources")
plt.show()

# Try to find real components with shiftNMF:
X = torch.tensor(X)
shiftnmf = ShiftNMF(X, 3)
W_, H_, tau_ = shiftnmf.run(verbose=True)
print(W_)
print(W)
# Plot the signals found by shiftNMF
plt.figure()
for signal in H_:
    plt.plot(signal)
plt.title("Signals determined by shiftNMF")
plt.show()

