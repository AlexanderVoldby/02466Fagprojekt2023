import numpy as np
import torch
import matplotlib.pyplot as plt
from shifted_dataset import plot_latent_components, plot_matrix, plot_data, gauss
from helpers.callbacks import explained_variance, NMI

def shift_NMF_dataset(W, H, tau):
    M = H.shape[1]
    W, H, tau = torch.Tensor(W), torch.Tensor(H), torch.Tensor(tau)
    # Fourier transform of H along the second dimension
    Hf = torch.fft.fft(H, dim=1)
    f = torch.arange(0, M) / M
    omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', tau, f))
    Wf = torch.einsum('Nd,NdM->NdM', W, omega)
    # Broadcast Wf and H together
    Vf = torch.einsum('NdM,dM->NM', Wf, Hf)
    V = torch.fft.ifft(Vf)
    return V.numpy()

def shift_AA_dataset(data, C, S, tau):
    X = torch.Tensor(data)
    C = torch.Tensor(C).type(torch.cdouble)
    S = torch.Tensor(S).type(torch.cdouble)
    tau = torch.Tensor(tau).type(torch.cdouble)
    N, M = X.shape
    # Implementation of shift AA.
    f = torch.arange(0, M) / M
    # first matrix Multiplication
    omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', tau, f))
    # omega_neg = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau()*(-1), f))
    omega_neg = torch.conj(omega)

    # data to frequency domain
    X_F = torch.fft.fft(X)

    # Aligned data (per component)
    X_F_align = torch.einsum('NM,NdM->NdM', X_F, omega_neg)
    # X_align = torch.fft.ifft(X_F_align)
    # The A matrix, (d,M) A, in frequency domain
    # self.A = torch.einsum('dN,NdM->dM', self.C(), X_align)
    # A_F = torch.fft.fft(self.A)
    A_F = torch.einsum('dN,NdM->dM', C, X_F_align)
    # S_F = torch.einsum('Nd,NdM->NdM', self.S().double(), omega)

    # archetypes back shifted
    # A_shift = torch.einsum('dM,NdM->NdM', self.A_F.double(), omega.double())
    S_shift = torch.einsum('Nd,NdM->NdM', S, omega)

    # Reconstruction
    components = torch.fft.ifft(A_F)
    xf = torch.einsum('NdM,dM->NM', S_shift, A_F)
    x = torch.fft.ifft(xf)

    return x.numpy(), components.numpy()


dir = "Results/shifted_dataset/"
shift_H = np.loadtxt(dir+"shiftNMF_disc_H.txt")
shift_W = np.loadtxt(dir+"shiftNMF_disc_W.txt")
shift_tau = np.loadtxt(dir+"shiftNMF_disc_tau.txt", dtype=np.complex_)
shift_tau = shift_tau.real
shift_C = np.loadtxt(dir+"shiftAA_disc_C.txt")
shift_S = np.loadtxt(dir+"shiftAA_disc_S.txt")
shift_AA_tau = np.loadtxt(dir+"shiftAA_disc_tau.txt", dtype=np.complex_)
shift_AA_tau = shift_AA_tau.real

AA_C = np.loadtxt(dir+"AA_C.txt")
AA_S = np.loadtxt(dir+"AA_S.txt")

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
softplus = torch.nn.Softplus()
softplus_W = softplus(torch.Tensor(W)).numpy()
data = shift_NMF_dataset(softplus_W, H, tau).real

AA_components = np.matmul(AA_C, data)

shift_AA_recon, shiftAA_comp = shift_AA_dataset(data, shift_C, shift_S, shift_AA_tau)
shift_NMF_recon = shift_NMF_dataset(shift_W, shift_H, shift_tau)

exp_var_shiftNMF = explained_variance(data, shift_NMF_recon.real)
exp_var_shiftAA = explained_variance(data, shift_AA_recon.real)
exp_var_NMF = explained_variance(data, np.matmul(NMF_W, NMF_H))
exp_var_AA = explained_variance(data, np.matmul(AA_S, AA_components))
print("Explained variance")
print(f"ShiftNMF: {exp_var_shiftNMF}")
print(f"ShiftAA: {exp_var_shiftAA}")
print(f"NMF: {exp_var_NMF}")
print(f"AA: {exp_var_AA}")
print(f"Normalized mutual information with the mixing matrices")
print(f"ShiftNMF: {NMI(W, shift_W)}")
print(f"ShiftAA: {NMI(W, shift_S)}")
print(f"NMF: {NMI(W, NMF_W)}")
print(f"AA: {NMI(W, AA_S)}")

plot_data(H, "Ground truth signals")
plot_matrix(W, "The mixing")
plot_matrix(tau, "The shifts")
plot_data(data, "Shifted and mixed dataset")

plot_latent_components(shift_H, ["1", "2", "3"], "Latent components found by shiftNMF")
plot_data(shift_NMF_recon, "Dataset reconstructed by shiftNMF")

plot_latent_components(shiftAA_comp, ["1", "2", "3"], "Latent components found by shiftAA")
plot_data(shift_AA_recon.real, "Dataset reconstructed by shiftAA")

plot_latent_components(NMF_H, ["1", "2", "3"], "Latent components found by NMF")
plot_data(np.matmul(NMF_W, NMF_H), "Reconstruction by NMF")

plot_latent_components(AA_components, ["1", "2", "3"], "Archetypes found by AA")
plot_data(np.matmul(AA_S, AA_components), "Reconstruction by AA")
