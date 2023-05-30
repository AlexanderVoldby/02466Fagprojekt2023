import scipy.io
import numpy as np
from torchNMF import NMF
from torchAA import torchAA
from sklearn import metrics
import matplotlib.pyplot as plt


def explained_variance(original_data, reconstructed_data):
    """
    Calculate the explained variance between original and reconstructed data.

    Args:
        original_data (numpy.ndarray): The original dataset.
        reconstructed_data (numpy.ndarray): The reconstructed dataset.

    Returns:
        float: The explained variance score.
    """
    numerator = np.sum(np.square(original_data - reconstructed_data))
    denominator = np.sum(np.square(original_data - np.mean(original_data)))
    explained_variance = 1 - (numerator / denominator)

    return explained_variance


# load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

# Get X and Labels. Probably different for the other dataset, but i didn't check :)
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

plt.figure()
plt.plot(X[[np.all(x == [35, 35, 30]) for x in targets]][0])
plt.xlabel("Chemical shift [ppm]")
plt.title("NMR spectrum of 35% ethanol, 35% butanol and 30% pentanol")
plt.show()

# Plot of the three pure components
fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].plot(X[[np.all(x == [100, 0, 0]) for x in targets]][0], label="Propanol")
axs[0].set_title("Propanol")
axs[1].plot(X[[np.all(x == [0, 100, 0]) for x in targets]][0], label="Butanol")
axs[1].set_title("Butanol")
axs[2].plot(X[[np.all(x == [0, 0, 100]) for x in targets]][0], label="Pentanol")
axs[2].set_title("Pentanol")
plt.show()

# Fit NMF and AA 10 times and get an average loss curve for each as well as an average explained variance
max_components = 10
aa_explained = []
nmf_explained = []
for i in range(max_components):
    print(F"Components: {i+1}")
    nmf = NMF(X, rank=i+1)
    aa = torchAA(X, rank=i+1)
    nmf.fit(verbose=True)
    aa.fit(verbose=True)
    nmf_explained.append(explained_variance(X, nmf.forward().detach().numpy()))
    aa_explained.append(explained_variance(X, aa.forward().detach().numpy()))

plt.figure()
plt.plot(np.arange(1, max_components+1), nmf_explained, label="NMF")
plt.plot(np.arange(1, max_components+1), aa_explained, label="AA")
plt.title("Explained variance per no. components used in factorization")
plt.legend()
plt.show()
plt.savefig("Explained_variance.png", format="png")

n_iterations = 10
nmf_W = np.empty(n_iterations)
nmf_H = np.empty(n_iterations)
aa_C = np.empty(n_iterations)
aa_S = np.empty(n_iterations)
nmf_loss = np.empty(n_iterations)
aa_loss = np.empty(n_iterations)
for i in range(n_iterations):
    print(f"Iteration {i+1}")
    nmf = NMF(X, 3)
    aa2 = torchAA(X, 3)
    nmf_W[i], nmf_H[i], nmf_loss[i] = nmf.fit(verbose=True, return_loss=True)
    aa_C[i], aa_S[i], aa_loss[i] = aa2.fit(verbose=True, return_loss=True)

min_nmf_length = min(len(row) for row in nmf_loss)
min_aa_length = min(len(row) for row in aa_loss)
nmf_loss = nmf_loss[:, min_nmf_length]
aa_loss = aa_loss[:, min_aa_length]
y1 = np.mean(nmf_loss, axis=0)
y2 = np.mean(aa_loss, axis=0)
x = np.arange(len(y1))
y1_std = np.std(nmf_loss, axis=0)
y2_std = np.std(aa_loss, axis=0)

plt.figure()
plt.plot(x, y1, label="NMF")
plt.fill_between(x, y1-y1_std, y1+y1_std, alpha=0.3)
plt.plot(x, y2, label="AA")
plt.fill_between(x, y2-y2_std, y2+y2_std, alpha=0.3)
plt.xlabel("Iteration")
plt.ylabel("Running squared loss of the factorizatins")
plt.legend()
plt.savefig("Running_loss.png", format="png")
