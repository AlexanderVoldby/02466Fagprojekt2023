import numpy as np
import matplotlib.pyplot as plt
from ShiftNMF_half_frequencies import ShiftNMF
import torch
from helpers.callbacks import explained_variance, ChangeStopper
from torchNMF import NMF, MVR_NMF
np.random.seed(42069)

# Define random sources, mixings and shifts; H, W and tau
N, M, d = 10, 10000, 3
Fs = 1000  # The sampling frequency we use for the simulation
t0 = 10     # The half-time interval we look at
t = np.arange(-t0, t0, 1/Fs)  # the time samples
f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

def plot_data(X, title=""):
    plt.figure()
    for signal in X:
        plt.plot(signal)
    plt.title(title)
    plt.show()

def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

def shift_dataset(W, H, tau):
    M = H.shape[1]
    W, H, tau = torch.Tensor(W), torch.Tensor(H), torch.Tensor(tau)
    # Get half of the frequencies
    Nf = M // 2 + 1
    # Fourier transform of H along the second dimension
    Hf = torch.fft.fft(H, dim=1)[:, :Nf] # Keep only the first Nf elements of the Fourier transform of H
    Hf_reverse = torch.flip(Hf[:, 1: Nf - 1], dims=[1])
    # Concatenate the original columns with the reversed columns along the second dimension
    Hft = torch.cat((Hf, torch.conj(Hf_reverse)), dim=1)
    # Construct the datamatrix by shifting and mixing
    f = torch.arange(0, M) / M
    omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', tau, f))
    Wf = torch.einsum('Nd,NdM->NdM', W, omega)
    # Broadcast Wf and H together
    Vf = torch.einsum('NdM,dM->NM', Wf, Hft)
    V = torch.fft.ifft(Vf)
    return V.numpy()

# Random mixings:
W = np.random.rand(N, d)
# Random gaussian shifts
tau = np.random.randint(-1000, 1000, size=(N, d))
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 50]
H = np.array([gauss(m, s, np.arange(0, 1000, 0.1)) for m, s in list(zip(mean, std))])

plot_data(H, "Ground truth signals")

X = shift_dataset(W, H, tau)
plot_data(X, "Dataset build from mixing and shifts of the three sources")

# Try to find real components with shiftNMF:
shiftnmf_loss = []
iterations = 1
models = [ShiftNMF(X, 3, alpha=1e-9) for i in range(iterations)]
params = []
for i in range(iterations):
    W_, H_, tau_, loss = models[i].fit(verbose=True, return_loss=True)
    shiftnmf_loss.append(loss[-1])
    params.append((W_, H_, tau_))
best_model = models[np.argmin(shiftnmf_loss)]
best_W, best_H, best_tau = params[np.argmin(shiftnmf_loss)]

plot_data(best_H, "Signals determined by shiftNMF")
plot_data(best_model.recon.detach().numpy(), "Dataset reconstructed by shiftNMF")


# Then with regular NMF:
nmf = NMF(X, 3)
Wnmf, Hnmf = nmf.fit(verbose=True)
recon_nmf = nmf.forward().detach().numpy()

plot_data(Hnmf, "Signals determined by NMF")
plot_data(recon_nmf, "Reconstruction by NMF")

