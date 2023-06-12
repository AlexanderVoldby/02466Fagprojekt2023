import numpy as np
import matplotlib.pyplot as plt
from shifted_dataset import plot_latent_components, plot_matrix, plot_data, shift_dataset, gauss
from helpers.callbacks import explained_variance


dir = "Results/shifted_dataset/"
shift_H = np.loadtxt(dir+"shiftNMF_disc_H.txt")
shift_W = np.loadtxt(dir+"shiftNMF_disc_W.txt")
shift_tau = np.loadtxt(dir+"shiftNMF_disc_tau.txt", dtype=np.complex_)
shift_tau = shift_tau.real
NMF_H = np.loadtxt(dir+"NMF_H.txt")
NMF_W = np.loadtxt(dir+"NMF_W.txt")

# Artificial data
N, M, d = 10, 10000, 3
W = np.random.randn(N, d)
# Random gaussian shifts
tau = np.round(np.random.randn(N, d)*100)
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 50]
H = np.array([gauss(m, s, np.arange(0, 1000, 0.1)) for m, s in list(zip(mean, std))])
data = shift_dataset(W, H, tau).real

exp_var_shiftNMF = explained_variance(data, shift_dataset(shift_W, shift_H, shift_tau).real)
exp_var_NMF = explained_variance(data, np.matmul(W, H))
print("Explained variance")
print(f"ShiftNMF: {exp_var_shiftNMF}")
print(f"NMF: {exp_var_NMF}")

plot_data(H, "Ground truth signals")
plot_matrix(W, "The mixing")
plot_matrix(tau, "The shifts")
plot_data(shift_dataset(W, H, tau), "Shifted and mixed dataset")

plot_latent_components(shift_H, ["1", "2", "3"], "Latent components found by shiftNMF")
plot_data(shift_dataset(shift_W, shift_H, shift_tau), "Dataset reconstructed by shiftNMF")

plot_latent_components(NMF_H, ["1", "2", "3"], "Latent components found by NMF")
plot_data(np.matmul(NMF_W, NMF_H), "Reconstruction by NMF")
