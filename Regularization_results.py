import numpy as np
import pickle
import matplotlib.pyplot as plt

# regs = np.logspace(-20, 0, 20)
regs = np.logspace(-50, -30, num=40)

def plot_components(matrix, title):
    fig, ax = plt.subplots(1, matrix.shape[0])
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i])
    plt.suptitle(title)
    plt.show()

# exp_var_L1 = []
# hs1 = []
# ws1 = []
# for reg in regs:
#     # Read the lists from "regularization_L1" file
#     with open(f"Results/Regularization/L1_reg_{reg}", "rb") as f:
#         regs1, exp_var, h, w = pickle.load(f)
#
#     exp_var_L1.append(exp_var)
#     hs1.append(h)
#     ws1.append(w)

# exp_var_L1 = []
# hs1 = []
# ws1 = []
# for i in range(len(regs)):
#     # Read the lists from "regularization_L1" file
#     with open(f"Results/Regularization3/L1_reg_{i}", "rb") as f:
#         reg, ev = pickle.load(f)
#
#     exp_var_L1.append(ev)


exp_var_L2 = []
hs2 = []
ws2 = []
# Read the lists from "regularization_L2" file
# for reg in regs:
#     with open(f"Results/Regularization/L2_reg_{reg}", "rb") as fb:
#         regs2, exp_var, h, w = pickle.load(fb)
#
#         exp_var_L2.append(exp_var)
#         hs2.append(h)
#         ws2.append(w)

for i in range(len(regs)):
    with open(f"Results/Regularization6/L2_reg_{i}", "rb") as fb:
        reg, ev, w, h = pickle.load(fb)
        exp_var_L2.append(ev)


# fig, ax = plt.subplots()
# ax.plot(regs, exp_var_L1, marker="o", linestyle="--")
# ax.set_xlabel("Regularization Strength")
# ax.set_ylabel("Explained variance using L1 normalization")
# ax.set_xscale("log")
# plt.show()

fig, ax = plt.subplots()
ax.plot(regs, exp_var_L2, marker="o", linestyle="--")
ax.set_xlabel("Regularization Strength")
ax.set_ylabel("Explained variance using L2 normalization")
ax.set_xscale("log")
plt.show()

# for h, reg in zip(hs_1, regs1):
#     plot_components(h, f"Latent components with regularization {reg}")
#
# for h, reg in zip(hs_2, regs2):
#     plot_components(h, f"Latent components with regularization {reg}")