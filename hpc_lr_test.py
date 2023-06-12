import torch
from Artificial_no_shift import X
X_art = X
import scipy

# load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

# Get X and Labels. Probably different for the other dataset, but i didn't check :)
X = mat.get('xData')
targets = mat.get('yData')
target_labels = mat.get('yLabels')
axis = mat.get("Axis")
X_alko = X
from torchNMF import NMF
from torchAA import torchAA
from torchShiftAA import torchShiftAA
from ShiftNMF_half_frequencies import ShiftNMF
import numpy as np
import matplotlib.pyplot as plt
from helpers.data import X_clean
X_oil = X_clean
from helpers.data import X
X_urine = X

print("starting")
model_name = "NMF"
data_name = "urine"
X = X_urine

lrs = [10**x for x in range(-5,1)]
nr_tests = 10
losses = np.zeros((len(lrs),nr_tests))

for i, lr in enumerate(lrs):
    print("learning rate:" + str(lr))
    for it in range(nr_tests):
        model = NMF(X, 3, lr=lr, factor=1, patience=10)
        returns = model.fit(verbose=False, return_loss=True)
        loss = returns[-1]
        losses[i,it] = loss[-1]

print(lrs)
print(np.mean(losses,axis=1))
print("all losses")
print(losses)
plt.ylabel("average loss")
plt.xlabel("Learning rate")
plt.plot([str(lr) for lr in lrs], np.mean(losses,axis=1).flatten())
plt.suptitle('Categorical Plotting')
plt.savefig("lr_test_"+str(model_name))


