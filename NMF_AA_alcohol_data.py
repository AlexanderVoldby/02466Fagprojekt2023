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
# nmf = torchNMF(X, 3)
# W, H = nmf.fit(verbose=True)

# aa = torchAA(X, 3)
# C, S = aa.fit(verbose=True)
