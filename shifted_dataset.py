import numpy as np
import matplotlib.pyplot as plt
from ShiftNMF_half_frequencies import ShiftNMF
from helpers.callbacks import explained_variance
from torchNMF import NMF, MVR_NMF
np.random.seed(42069)

# Define random sources, mixings and shifts; H, W and tau
N, M, d = 10, 10000, 3
Fs = 1000  # The sampling frequency we use for the simulation
t0 = 10     # The half-time interval we look at
t = np.arange(-t0, t0, 1/Fs)  # the time samples
f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)


def shift_dataset(W, H, tau):
    # Get half the frequencies
    Nf = H.shape[1] // 2 + 1
    # Fourier transform of S along the second dimension
    Hf = np.fft.fft(H, axis=1)
    # Keep only the first Nf[1] elements of the Fourier transform of S
    Hf = Hf[:, :Nf]
    # Construct the shifted Fourier transform of S
    Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
    # Concatenate the original columns with the reversed columns along the second dimension
    Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
    f = np.arange(0, M) / M
    omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
    Wf = np.einsum('Nd,NdM->NdM', W, omega)
    # Broadcast Wf and H together
    Vf = np.einsum('NdM,dM->NM', Wf, Hft)
    V = np.fft.ifft(Vf)
    return V

# Random mixings:
W = np.random.rand(N, d)
# Random gaussian shifts
tau = np.random.randint(0, 1000, size=(N, d))
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 7]
t = np.arange(0, 1000, 0.1)
H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])
plt.figure()
for signal in H:
    plt.plot(signal)
plt.title("Original signals")
plt.show()

X = shift_dataset(W, H, tau)

plt.figure()
for signal in X:
    plt.plot(signal.real)
plt.title("Dataset build from mixing and shifts of the three sources")
plt.show()

# Try to find real components with shiftNMF:
# shiftnmf = ShiftNMF(X, 3)
# W_, H_, tau_ = shiftnmf.fit(verbose=True)
# # Plot the signals found by shiftNMF
# plt.figure()
# for signal in H_:
#     plt.plot(signal)
# plt.title("Signals determined by shiftNMF")
# plt.show()

# Then with regular NMF:
nmf = NMF(X, 3)
Wnmf, Hnmf = nmf.fit(verbose=True)
plt.figure()
for signal in Hnmf:
    plt.plot(signal)
plt.title("Signals determined by NMF")
plt.show()
