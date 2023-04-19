import scipy.io
import numpy as np
from torchNMF import torchNMF
from torchAA import torchAA
import matplotlib.pyplot as plt

#load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

#Get X and Labels. Probably different for the other dataset, but i didn't check :)
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")

# Plot some signal from the data
plt.figure()
for signal in X[:10]:
    plt.plot(signal)
plt.title("First 10 signals")
plt.show()

# Fit NMF with 3 components corresponding to the 3 alcohol types
nmf = torchNMF(X, 3)
W, H = nmf.fit(verbose=True)

# Plot the signals found by NMF
plt.figure()
for component in H:
    plt.plot(component)
plt.show()
