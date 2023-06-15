import scipy
import numpy as np
import matplotlib.pyplot as plt
from helpers.callbacks import explained_variance, plot_data, train_n_times
import pickle
from torchNMF import MVR_NMF



def plot_components(matrix, title):
    fig, ax = plt.subplots(1, matrix.shape[0])
    for i in range(matrix.shape[0]):
        ax[i].plot(matrix[i])
    plt.suptitle(title)
    plt.show()

def regularization_experiment(lower, upper, X, reg_mode, n_train=3, components=3):
    regs = np.logspace(lower, upper, num=2*abs(lower-upper), endpoint=False)
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

regs1, exp_var_1, hs_1, ws_1 = regularization_experiment(-42, -30, X, 1)
regs2, exp_var_2, hs_2, ws_2 = regularization_experiment(-42, -30, X, 2)

# Save regs1 and exp_var_1 to "regularization_L1" file
with open("Results/Regularization/regularization_L1", "wb") as f:
    pickle.dump((regs1, exp_var_1, hs_1, ws_1), f)

# Save regs2 and exp_var_2 to "regularization_L2" file
with open("Results/Regularization/regularization_L2", "wb") as fb:
    pickle.dump((regs2, exp_var_2, hs_2, ws_2), fb)

