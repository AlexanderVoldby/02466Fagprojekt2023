import scipy
import numpy as np
import torch
import matplotlib.pyplot as plt
from helpers.callbacks import explained_variance, plot_data
from torchNMF import MVR_NMF

mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

# Run MVR on a finer grid and plot it
regs = np.logspace(-8, 2, num=10, endpoint=False)
explained_vars = []
hs = []
for reg in regs:
    mvr_nmf = MVR_NMF(X, 3, reg)
    W, H = mvr_nmf.fit(verbose=True)
    recon = mvr_nmf.forward().detach().numpy()
    explained_vars.append(explained_variance(X, recon))
    hs.append(H)
print(explained_vars)

fig, ax = plt.subplots()
ax.plot(regs, explained_vars, marker="o", linestyle="--")
ax.set_xlabel("Regularization Strength")
ax.set_ylabel("Explained variance")
ax.set_xscale("log")
plt.show()

for h, reg in zip(hs, regs):
    plot_data(h, f"Latent components with regularization {reg}")
