import scipy.io
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from helpers.callbacks import explained_variance, plot_data
import matplotlib.pyplot as plt


def plot_latent_components(matrix, labs, title):
    n = matrix.shape[0]
    fig, axs = plt.subplots(1, n, sharey=True)
    for i in range(n):
        axs[i].plot(matrix[i])
        axs[i].set_title(labs[i])

    fig.suptitle(title)
    plt.show()

def component_plot(components, labels, title):
    x = np.arange(components.shape[0])
    plt.figure(figsize=(10, 6))  # Adjust the figure size as per your preference
    plt.bar(x, components[:, 0], label=labels[0])
    plt.bar(x, components[:, 1], bottom=components[:, 0], label=labels[1])
    plt.bar(x, components[:, 2], bottom=components[:, 0] + components[:, 1], label=labels[2])


    # Adding labels and titles
    plt.xlabel('Samples')
    plt.ylabel('Percentage')
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()


def sort_rows_by_squared_error(true_matrix, other_matrix):
    # Compute the squared error between each row in `other_matrix` and `true_matrix`
    mutual_infos = np.zeros((other_matrix.shape[0], true_matrix.shape[0]))
    for i in range(other_matrix.shape[0]):
        mutual_infos[i] = mutual_info_regression(true_matrix.T,
                                                    other_matrix[i],
                                                    discrete_features=False)

    # Get the indices that would sort the rows of `other_matrix` based on squared error
    # Remove all except the most similar index
    sorted_indices = np.argsort(mutual_infos, axis=1)[:, -1]

    # Create a sorted version of `other_matrix` based on the computed indices
    sorted_other_matrix = other_matrix[sorted_indices]

    return sorted_other_matrix, sorted_indices


# load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

# Get X and Labels. Probably different for the other dataset, but i didn't check :)
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

W = np.loadtxt("Results/W_alcohol.txt")
H = np.loadtxt("Results/H_alcohol.txt")
C = np.loadtxt("Results/C_alcohol.txt")
S = np.loadtxt("Results/S_alcohol.txt")
aa_features = np.matmul(C, X)

print(f"Explained variance of NMF reconstruction: {explained_variance(X, np.matmul(W, H))}")
print(f"Explained variance of AA reconstruction: {explained_variance(X, np.matmul(S, aa_features))}")

labels = [[100, 0, 0], [0, 100, 0], [0, 0, 100]]
alcohols = np.array([X[[np.all(x == label) for x in targets]][0] for label in labels])
H, H_indx = sort_rows_by_squared_error(alcohols, H)
W = W.T[H_indx].T
aa_features, aa_indx = sort_rows_by_squared_error(alcohols, aa_features)
S = S.T[aa_indx].T

plot_latent_components(alcohols, target_labels, "Pure alcohols")
plot_latent_components(H, target_labels, "NMF latent components")
plot_latent_components(aa_features, target_labels, "AA latent components")

W_fraction = (W / np.sum(W, axis=1).reshape(W.shape[0], 1)) * 100
S_fraction = (S / np.sum(S, axis=1).reshape(S.shape[0], 1)) * 100

component_plot(W_fraction, target_labels, "Concentrations estimated with NMF")
component_plot(S_fraction, target_labels, "Concentrations estimated with AA")
component_plot(targets, target_labels, "True concentrations")

