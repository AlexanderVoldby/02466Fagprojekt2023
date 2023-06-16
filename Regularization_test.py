import scipy
import sys
import numpy as np
import matplotlib.pyplot as plt
from helpers.callbacks import explained_variance, plot_data, train_n_times
import pickle
from torchNMF import MVR_NMF

lower = -42
upper = -30
regs = np.logspace(lower, upper, num=2*abs(lower-upper), endpoint=False)
num_regs = len(regs)
x = sys.argv[1]

def plot_components(matrix, title):
    fig, ax = plt.subplots(1, matrix.shape[0])
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i])
    plt.suptitle(title)
    plt.show()

def regularization_experiment(reg, X, reg_mode, n_train=5, components=3):
    # print(f"Current regularization strength: {reg}")
    (W, H), loss = train_n_times(n_train, MVR_NMF, X, components, regularization=reg, normalization=reg_mode)
    recon = np.matmul(W, H)
    explained_var = explained_variance(X, recon)

    return reg, explained_var, W, H


mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

reg1, exp_var_1, w1, h1 = regularization_experiment(regs[x], X, 1)
reg2, exp_var_2, w2, h2 = regularization_experiment(regs[x], X, 2)

# Save regs1 and exp_var_1 to "regularization_L1" file
with open(f"Results/Regularization/L1_reg_{reg1}", "wb") as f:
    pickle.dump((reg1, exp_var_1, w1, h1), f)

# Save regs2 and exp_var_2 to "regularization_L2" file
with open(f"Results/Regularization/L2_reg_{reg2}", "wb") as fb:
    pickle.dump((reg2, exp_var_2, w2, h2), fb)

