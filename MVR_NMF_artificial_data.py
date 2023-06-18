import numpy as np
from helpers.callbacks import explained_variance, train_n_times
import matplotlib.pyplot as plt
from torchNMF import MVR_NMF
import pickle
from shifted_dataset import plot_data

def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

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

N, M, d = 10, 10000, 3
W = np.random.dirichlet(np.ones(d), N)
mean = [40, 300, 700]
std = [10, 20, 7]
t = np.arange(0, 1000, 0.1)
H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])
data = np.matmul(W, H)
print(f"Determinant volume approximation: {np.linalg.det(np.matmul(H, H.T))}")

regs = [1e-40, 1e-30, 1e-20, 1e-10, 1e-5, 1]
expvar = []
for reg in regs:
    print(f"Fitting reg = {reg}")
    (W, H), loss = train_n_times(5, MVR_NMF, data, 3, regularization=reg)
    recon = np.matmul(W, H)
    ev  = explained_variance(data, recon)
    expvar.append(ev)
    print(f"Explained variance: {ev}")

with open("Results/Regularization/artificial", "wb") as f:
    pickle.dump((regs, expvar), f)

# with open("Results/Regularization/artificial", "rb") as f:
#     pickle.load((regs, expvar), f)
