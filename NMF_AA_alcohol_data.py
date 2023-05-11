import scipy.io
import numpy as np
from torchNMF import NMF
from torchAA import torchAA
from sklearn import metrics
import matplotlib.pyplot as plt


def explained_variance(latent_var, mixing, data, scorer=metrics.explained_variance_score):
    prediction = np.matmul(mixing, latent_var)
    return scorer(data, prediction)

#load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

#Get X and Labels. Probably different for the other dataset, but i didn't check :)
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
plt.figure()
plt.plot(X[[np.all(x == [100, 0, 0]) for x in targets]][0], label="Propanol")
plt.plot(X[[np.all(x == [0, 100, 0]) for x in targets]][0], label="Butanol")
plt.plot(X[[np.all(x == [0, 0, 100]) for x in targets]][0], label="Pentanol")
plt.legend()
plt.xlabel("Chemical shift [ppm]")
plt.title("Spectrums of the three pure alcohols")
plt.show()

# Fit NMF with 3 components corresponding to the 3 alcohol types
nmf = NMF(X, 3)
W, H = nmf.fit(verbose=True)
print(f"Expained variance by NMF (1 run): {explained_variance(H, W, X)}")
# Plot the signals and the mixings found by NMF
plt.figure()
for signal in H:
    plt.plot(signal)
plt.title("Signals found by NMF")
plt.show()

# Same but with AA
aa = torchAA(X, 3)
C, S = aa.fit(verbose=True)
print(f"Expained variance by AA (1 run): {explained_variance(np.matmul(C, X), S, X)}")
archetypes = np.matmul(C, X)
for signal in archetypes:
    plt.plot(signal)
plt.title("Archetypes found by AA")
plt.show()
