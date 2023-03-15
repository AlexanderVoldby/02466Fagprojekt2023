from torchNMFtopy import torchNMF
# from torchAAtopy import torchAA
import numpy as np
import torch

# Create artificial 3d dataset of 4 gaussian clusters
m1 = [3., 3., 3.]
m2 = [4., 7., 3.]
m3 = [5., 3., 3.]
m4 = [5., 5., 8.]
sigma = 3
cov = np.eye(3)*sigma

X = np.array([np.random.multivariate_normal(m, cov, size=25) for m in [m1, m2, m3, m4]])
# reshape data into 100 x 3 data matrix
X = X.reshape((4*25, 3))

# Run NMF, AA and min-volume NMF and plot

nmf = torchNMF(X, 4)
W, H = nmf.run(verbose=True)
print(W)
