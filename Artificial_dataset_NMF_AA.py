from torchNMFtopy import torchNMF
from NMF_vol_min import TorchNMF_MinVol
# from torchAAtopy import torchAA
import numpy as np
import matplotlib.pyplot as plt
import torch

# Sparseness function for investigating H


# Create artificial 3d dataset of 4 gaussian clusters
m1 = np.random.uniform(low=0.0, high=10, size=3) + 10
m2 = np.random.uniform(low=0.0, high=10, size=3) + 10
m3 = np.random.uniform(low=0.0, high=10, size=3) + 10
sigma = 7
cov = np.eye(3)*sigma
clusters = np.array([np.random.multivariate_normal(m, cov, size=25) for m in [m1, m2, m3]])
# reshape data into 100 x 3 data matrix
X = clusters.reshape((3*25, 3))
mean = np.mean([m1, m2, m3], axis=0)
mean_len = np.sqrt(sum([i**2 for i in mean]))
# Run NMF, AA and min-volume NMF and plot

nmf = torchNMF(X.T, 3)
W, H = nmf.run(verbose=True)
print(W.shape, H.shape)

nmf_min_vol = TorchNMF_MinVol(X.T, 3)
W_mv, H_mv = nmf_min_vol.run(verbose=True)
print(W_mv.shape, H_mv.shape)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ["g", "b", "r", "orange"]
for cluster, color in zip(clusters, colors):
    xs = cluster[:, 0]
    ys = cluster[:, 1]
    zs = cluster[:, 2]
    ax.scatter(xs, ys, zs, color=color)

for vec in W.T:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    ax.quiver(0, 0, 0, x*mean_len, y*mean_len, z*mean_len, color="r", ls="--")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ["g", "b", "r", "orange"]
for cluster, color in zip(clusters, colors):
    xs = cluster[:, 0]
    ys = cluster[:, 1]
    zs = cluster[:, 2]
    ax.scatter(xs, ys, zs, color=color)

for vec in W_mv.T:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    ax.quiver(0, 0, 0, x*mean_len, y*mean_len, z*mean_len, color="r", ls="--")

plt.show()

