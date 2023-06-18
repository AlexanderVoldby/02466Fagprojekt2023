import numpy as np
import matplotlib.pyplot as plt
from ShiftNMFDiscTau import ShiftNMF
from torchShiftAADiscTau import torchShiftAADisc
import torch
from torchAA import torchAA
from helpers.callbacks import explained_variance, ChangeStopper, plot_data, train_n_times
from torchNMF import NMF, MVR_NMF
np.random.seed(42069)

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


def plot_latent_components(matrix, labs, title):
    n = matrix.shape[0]
    fig, axs = plt.subplots(1, n, sharey=True)
    for i in range(n):
        axs[i].plot(matrix[i])
        axs[i].set_title(labs[i])

    fig.suptitle(title)
    plt.show()
def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

def shift_dataset(W, H, tau):
    softplus = torch.nn.Softplus()
    M = H.shape[1]
    W, H, tau = torch.Tensor(W), torch.Tensor(H), torch.Tensor(tau)
    # Fourier transform of H along the second dimension
    Hf = torch.fft.fft(H, dim=1)
    f = torch.arange(0, M) / M
    omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', tau, f))
    Wf = torch.einsum('Nd,NdM->NdM', softplus(W), omega)
    # Broadcast Wf and H together
    Vf = torch.einsum('NdM,dM->NM', Wf, Hf)
    V = torch.fft.ifft(Vf)
    return V.numpy()

if __name__ == "__main__":
    plot = False
    # Define random sources, mixings and shifts; H, W and tau
    # Random mixings:
    N, M, d = 10, 10000, 3
    W = np.random.dirichlet(np.ones(d), N)
    W = np.append(W, [[1, 0, 0]], axis=0)
    W = np.append(W, [[0, 1, 0]], axis=0)
    W = np.append(W, [[0, 0, 1]], axis=0)
    N = N + 3

    # W = np.random.rand(N, d)
    # Random gaussian shifts
    tau = np.random.randint(-300, 300, (N-3, d))
    tau = np.vstack((tau, np.zeros((3, d))))
    # Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
    mean = [40, 300, 700]
    std = [10, 20, 7]
    t = np.arange(0, 1000, 0.1)
    H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)
    np.savetxt("Results/shifted_dataset_dirichlet/data.txt", X)
    np.savetxt("Results/shifted_dataset_dirichlet/mixing.txt", W)
    plot_data(H, "Ground truth signals")
    plot_matrix(W, "The mixing")
    plot_matrix(tau, "The shifts")
    plot_data(X, "Shifted and mixed dataset")

    (best_W, best_H, best_tau), loss = train_n_times(1, ShiftNMF, X, 3, alpha=1e-9, lr=0.1, min_imp=1e-9)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftNMF_disc_H.txt", best_H)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftNMF_disc_W.txt", best_W)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftNMF_disc_tau.txt", best_tau)

    (best_C, best_S, best_AA_tau), loss2 = train_n_times(1, torchShiftAADisc, X, 3, alpha=1e-9, lr=0.1, min_imp=1e-9)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftAA_disc_C.txt", best_C)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftAA_disc_S.txt", best_S)
    np.savetxt("Results/shifted_dataset_dirichlet/shiftAA_disc_tau.txt", best_AA_tau)

    # Then with regular NMF:
    # (nmfW, nmfH), loss2 = train_n_times(5, NMF, X, 3)
    # np.savetxt("Results/shifted_dataset/NMF_H.txt", nmfH)
    # np.savetxt("Results/shifted_dataset/NMF_W.txt", nmfW)

    # Then with regular AA:
    # (C, S), loss3 = train_n_times(5, torchAA, X, 3)
    # np.savetxt("Results/shifted_dataset/AA_C.txt", C)
    # np.savetxt("Results/shifted_dataset/AA_S.txt", S)

    if plot:
        pass
        # plot_latent_components(H, ["1", "2", "3"], "Latent components found by shiftNMF")
        # plot_data(shift_dataset(best_W, best_H, best_tau), "Dataset reconstructed by shiftNMF")

        # plot_latent_components(nmfH, ["1", "2", "3"], "Signals determined by NMF")
        # plot_data(np.matmul(nmfW, nmfH), "Reconstruction by NMF")