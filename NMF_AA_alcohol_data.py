import scipy.io
import numpy as np
from torchNMF import NMF
from torchAA import torchAA
from helpers.callbacks import explained_variance, ChangeStopper
from sklearn import metrics
import matplotlib.pyplot as plt


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
    nmf = NMF(X, i+1)
    aa = torchAA(X, i+1)
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
# plt.figure()
# plt.plot(X[[np.all(x == [35, 35, 30]) for x in targets]][0])
# plt.xlabel("Chemical shift [ppm]")
# plt.title("NMR spectrum of 35% ethanol, 35% butanol and 30% pentanol")
# plt.show()

# Plot of the three pure components
# fig, axs = plt.subplots(1, 3, sharey=True)
# axs[0].plot(X[[np.all(x == [100, 0, 0]) for x in targets]][0], label="Propanol")
# axs[0].set_title("Propanol")
# axs[1].plot(X[[np.all(x == [0, 100, 0]) for x in targets]][0], label="Butanol")
# axs[1].set_title("Butanol")
# axs[2].plot(X[[np.all(x == [0, 0, 100]) for x in targets]][0], label="Pentanol")
# axs[2].set_title("Pentanol")
# plt.show()
n_iterations = 1
nmf_W = np.empty(n_iterations, dtype=object)
nmf_H = np.empty(n_iterations, dtype=object)
aa_C = np.empty(n_iterations, dtype=object)
aa_S = np.empty(n_iterations, dtype=object)
nmf_loss = []
aa_loss = []
for i in range(n_iterations):
    print(f"Iteration {i+1}")
    nmf = NMF(X, 3)
    aa2 = torchAA(X, 3)
    nmf_W[i], nmf_H[i], loss1 = nmf.fit(verbose=True, return_loss=True)
    aa_C[i], aa_S[i], loss2 = aa2.fit(verbose=True, return_loss=True)
    nmf_loss.append(loss1)
    aa_loss.append(loss2)

min_nmf_length = min([len(row) for row in nmf_loss])
min_aa_length = min([len(row) for row in aa_loss])
print(min_nmf_length, min_aa_length)
nmf_loss = np.array(nmf_loss)[:, :min_nmf_length]
aa_loss = np.array(aa_loss)[:, :min_aa_length]
y1 = np.mean(nmf_loss, axis=0)
y2 = np.mean(aa_loss, axis=0)
x1 = np.arange(y1.size)
x2 = np.arange(y2.size)
y1_std = np.std(nmf_loss, axis=0)
y2_std = np.std(aa_loss, axis=0)

plt.figure()
plt.plot(x1, y1, label="NMF")
plt.fill_between(x1, y1-y1_std, y1+y1_std, alpha=0.3)
plt.plot(x2, y2, label="AA")
plt.fill_between(x2, y2-y2_std, y2+y2_std, alpha=0.3)
plt.xlabel("Iteration")
plt.ylabel("Running squared loss of the factorizatins")
plt.legend()

nmf_W_mean = np.mean(nmf_W, axis=0)
nmf_H_mean = np.mean(nmf_H, axis=0)
aa_C_mean = np.mean(aa_C, axis=0)
aa_S_mean = np.mean(aa_S, axis=0)

plt.figure()
for v in nmf_H_mean:
    plt.plot(v)
plt.title("Estimated components by NMF")
plt.show()

plt.figure()
for v in np.matmul(aa_C_mean, X):
    plt.plot(v)
plt.title("Estimated archetypes by AA")
plt.show()

# Get the fraction of each component in the rebuilt samples. This should be an estimate of the content of each alcohol in each sample.
W_fraction = nmf_W_mean / np.sum(nmf_W_mean, axis=1).reshape(nmf_W_mean.shape[0], 1)
S_fraction = aa_S_mean / np.sum(aa_S_mean, axis=1).reshape(aa_S_mean.shape[0], 1)
# SHow images of the matrices with colorbars
plt.matshow(W_fraction)
plt.colorbar()
plt.title("Estimated fractions of each component using NMF")

plt.matshow(S_fraction)
plt.colorbar()
plt.title("Estimated fractions of each component using AA")

plt.matshow(targets)
plt.colorbar()
plt.title("True fractions of each alcohol")
