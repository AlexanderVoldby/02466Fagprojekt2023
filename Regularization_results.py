import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt


# Read the lists from "regularization_L1" file
with open("Results/Regularization/regularization_L1", "rb") as f:
    regs1, exp_var_1, hs_1, ws_1 = pickle.load(f)

# Read the lists from "regularization_L2" file
with open("Results/Regularization/regularization_L2", "rb") as fb:
    regs2, exp_var_2, hs_2, ws_2 = pickle.load(fb)

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