import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.data import X
from shifted_dataset import plot_latent_components, plot_matrix, plot_data, gauss
from helpers.callbacks import explained_variance, NMI

def plot_matrix(mat, title):
    n, m = mat.shape
    plt.figure()
    plt.imshow(mat, aspect='auto', interpolation="none")
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_yticks(np.arange(0, n, 1))
    plt.title(title)
    plt.show()

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


dir = "Results/urine_data/"
shift_H = np.loadtxt(dir+"H_shift_nmf.txt")
shift_W = np.loadtxt(dir+"W_shift_nmf.txt")
shift_tau = np.loadtxt(dir+"tau_nmf.txt", dtype=np.complex_)
shift_tau = shift_tau
shift_C = np.loadtxt(dir+"C_shift_AA.txt")
shift_S = np.loadtxt(dir+"S_shift_AA.txt")
shift_AA_tau = np.loadtxt(dir+"tau_AA.txt", dtype=np.complex_)
shift_AA_tau = shift_AA_tau

AA_C = np.loadtxt(dir+"C_reg_AA.txt")
AA_S = np.loadtxt(dir+"S_reg_AA.txt")

NMF_H = np.loadtxt(dir+"H_reg_nmf.txt")
NMF_W = np.loadtxt(dir+"W_reg_nmf.txt")

# W = np.loadtxt(dir+"W_true.txt")
# H = np.loadtxt(dir+"H_true.txt")
# tau = np.loadtxt(dir+"true_tau.txt")

data = X

plot_data(data, "Urine dataset")
# plot_latent_components(H, ["1", "2", "3"], "True latent components")
# plot_matrix(W, "True mixing")
# plot_matrix(tau, "True shifts")

AA_comp = np.matmul(AA_C, data)
AA_recon = np.matmul(AA_S, AA_comp)
NMF_recon = np.matmul(NMF_W, NMF_H)

shift_AA_recon, shiftAA_comp = shift_AA_dataset(X, shift_C, shift_S, shift_AA_tau)
shift_NMF_recon = shift_NMF_dataset(shift_W, shift_H, shift_tau)

shift_AA_recon_no_shifts, shiftAA_no_shift_comp = shift_AA_dataset(X, shift_C, shift_S, np.zeros(shift_AA_tau.shape))
shift_NMF_recon_no_shifts = shift_NMF_dataset(shift_W, shift_H, np.zeros(shift_tau.shape))

exp_var_shiftNMF = explained_variance(data, shift_NMF_recon.real)
exp_var_shiftAA = explained_variance(data, shift_AA_recon.real)
exp_var_NMF = explained_variance(data, NMF_recon)
exp_var_AA = explained_variance(data, AA_recon)


print("Explained variance")
print(f"ShiftNMF: {exp_var_shiftNMF}")
print(f"ShiftAA: {exp_var_shiftAA}")
print(f"NMF: {exp_var_NMF}")
print(f"AA: {exp_var_AA}")

# print(f"Normalized mutual information with the mixing matrices")
# print(f"ShiftNMF: {NMI(W, shift_W)}")
# print(f"ShiftAA: {NMI(W, shift_S)}")
# print(f"NMF: {NMI(W, AA_S)}")
# print(f"AA: {NMI(W, NMF_W)}")

plot_latent_components(NMF_H, ["1", "2", "3"], "Latent components found by NMF")
plot_data(NMF_recon, "Dataset reconstructed by NMF")

plot_latent_components(AA_comp, ["1", "2", "3"], "Latent components found by AA")
plot_data(AA_recon, "Dataset reconstructed by AA")

plot_latent_components(shift_H, ["1", "2", "3"], "Latent components found by shiftNMF")
plot_data(shift_NMF_recon, "Dataset reconstructed by shiftNMF")
plot_matrix(shift_W, "Mixing determined by shiftNMF")
plot_matrix(shift_tau.real, "Shifts determined by shiftNMF")

plot_latent_components(shiftAA_comp, ["1", "2", "3"], "Latent components found by shiftAA")
plot_data(shift_AA_recon.real, "Dataset reconstructed by shiftAA")
plot_matrix(shift_S, "Mixing determined by shiftAA")
plot_matrix(shift_AA_tau.real, "Shifts determined by shiftAA")

plot_data(shift_NMF_recon_no_shifts, "Dataset reconstructed by shiftNMF without shifts")
plot_data(shift_AA_recon_no_shifts, "Dataset reconstructed by shiftAA without shifts")