import scipy
import numpy as np
import matplotlib.pyplot as plt
from helpers.callbacks import explained_variance, plot_data, train_n_times
from torchNMF import MVR_NMF

def plot_components(matrix, title):
    fig, ax = plt.subplots(1, matrix.shape[0])
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i])
    plt.suptitle(title)
    plt.show()

def regularization_experiment(lower, upper, X, reg_mode, n_train=1, components=3):
    regs = np.logspace(lower, upper, num=abs(lower-upper), endpoint=False)
    explained_vars = []
    hs = []
    ws = []
    for reg in regs:
        print(f"Current regularization strength: {reg}")
        (W, H), loss = train_n_times(n_train, MVR_NMF, X, components, regularization=reg, normalization=reg_mode)
        recon = np.matmul(W, H)
        explained_vars.append(explained_variance(X, recon))
        hs.append(H)
        ws.append(W)

    return regs, explained_vars, hs, ws


mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

regs1, exp_var_1, hs_1, ws_1 = regularization_experiment(-32, 22, X, 1)
regs2, exp_var_2, hs_2, vs_2 = regularization_experiment(-32, 22, X, 2)

fig, ax = plt.subplots()
ax.plot(regs1, exp_var_1, marker="o", linestyle="--")
ax.set_xlabel("Regularization Strength")
ax.set_ylabel("Explained variance using L1 normalization")
ax.set_xscale("log")
plt.show()

fig, ax = plt.subplots()
ax.plot(regs2, exp_var_2, marker="o", linestyle="--")
ax.set_xlabel("Regularization Strength")
ax.set_ylabel("Explained variance using L2 normalization")
ax.set_xscale("log")
plt.show()

for h, reg in zip(hs_1, regs1):
    plot_components(h, f"Latent components with regularization {reg}")

for h, reg in zip(hs_2, regs2):
    plot_components(h, f"Latent components with regularization {reg}")
