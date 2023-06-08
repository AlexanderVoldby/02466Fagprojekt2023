import scipy.io
import numpy as np
from torchNMF import NMF
from torchAA import torchAA
from helpers.callbacks import explained_variance, plot_data, train_n_times
import matplotlib.pyplot as plt

label_colors = {}
def component_plot(components, labels, title):
    global label_colors
    pos_colors = ["r", "g", "b"]
    k = 0
    for label in labels:
        if label not in label_colors:
            color = pos_colors[k]
            label_colors[label] = color
            k += 1
    x = np.arange(components.shape[0])
    plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
    plt.bar(x, components[:, 0], label=labels[0],
            color=label_colors[labels[0]])
    plt.bar(x, components[:, 1], bottom=components[:, 0], label=labels[1],
            color=label_colors[labels[1]])
    plt.bar(x, components[:, 2], bottom=components[:, 0] + components[:, 1], label=labels[2],
            color=label_colors[labels[2]])

    # Adding labels and titles
    plt.xlabel('Samples')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()

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
exp_var = False
if exp_var:
    max_components = 7
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

nmf_params, nmf_loss = train_n_times(5, NMF, X, 3, lr=0.2)
aa_params, aa_loss = train_n_times(10, torchAA, X, 3, lr=0.2)
W, H = nmf_params
C, S = aa_params
aa_components = np.matmul(C, X)
np.savetxt("W_alcohol.txt", W)
np.savetxt("H_alcohol.txt", H)
np.savetxt("C_alcohol.txt", C)
np.savetxt("S_alcohol.txt", S)

# Find the alcohols that the components correspond to with least squares difference
labels = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
alcohols = np.array([X[[np.all(x == label) for x in targets]][0] for label in labels])
alcohols_normalized = (alcohols - np.mean(alcohols, axis=0))/np.std(alcohols, axis=0)
H_norm = (H-np.mean(H, axis=0))/np.std(H, axis=0)
ac_norm = (aa_components-np.mean(aa_components, axis=0))/np.std(aa_components, axis=0)
NMF_comp_labels = [target_labels[np.argmin(np.sum(np.square(alcohols_normalized-component), axis=1))]
                   for component in H_norm]
AA_comp_labels = [target_labels[np.argmin(np.sum(np.square(alcohols_normalized-component), axis=1))]
                  for component in ac_norm]


fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].plot(H[0])
axs[0].set_title(NMF_comp_labels[0])
axs[1].plot(H[1])
axs[1].set_title(NMF_comp_labels[1])
axs[2].plot(H[2])
axs[2].set_title(NMF_comp_labels[2])
fig.suptitle("Latent components found by NMF")
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
axs[0].plot(aa_components[0])
axs[0].set_title(AA_comp_labels[0])
axs[1].plot(aa_components[1])
axs[1].set_title(AA_comp_labels[1])
axs[2].plot(aa_components[2])
axs[2].set_title(AA_comp_labels[2])
fig.suptitle("Archetypes found by AA")
plt.show()

# Get the fraction of each component in the rebuilt samples. This should be an estimate of the content of each alcohol in each sample.
W_fraction = (W / np.sum(W, axis=1).reshape(W.shape[0], 1)) * 100
S_fraction = (S / np.sum(S, axis=1).reshape(S.shape[0], 1)) * 100
# Show images of the matrices with colorbars

component_plot(W_fraction, NMF_comp_labels, "Contents of each component found by NMF")
component_plot(S_fraction, AA_comp_labels, "Contents of each component found by AA")
component_plot(targets, target_labels, "True contents")
