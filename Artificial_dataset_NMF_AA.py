from torchNMF import NMF, MVR_NMF
from NMF_vol_min import TorchNMF_MinVol
# from torchAAtopy import torchAA
import numpy as np
import matplotlib.pyplot as plt
import torch
np.random.seed(42069)
# Create artificial 3d dataset of 4 gaussian clusters
m1 = np.random.uniform(low=0.0, high=10, size=3) + 5
m2 = np.random.uniform(low=0.0, high=10, size=3) + 15
m3 = np.random.uniform(low=5.0, high=10, size=3) + 10
sigma = 7
cov = np.eye(3)*sigma
clusters = np.array([np.random.multivariate_normal(m, cov, size=25) for m in [m1, m2, m3]])
# reshape data into 100 x 3 data matrix
X = clusters.reshape((3*25, 3))
mean = np.mean([m1, m2, m3], axis=0)
mean_len = np.sqrt(sum([i**2 for i in mean]))
# Run NMF, AA and min-volume NMF and plot

nmf = NMF(X, 3) # X.T because W is a basis for the columns of X
W, H = nmf.fit(verbose=True)
print(W.shape, H.shape)
print(f"Volume of H: {np.linalg.det(H@H.T)}")

nmf_min_vol = MVR_NMF(X, 3)
W_mv, H_mv = nmf_min_vol.fit(verbose=True)
print(W_mv.shape, H_mv.shape)
print(f"Volume of H_mv: {np.linalg.det(H_mv@H_mv.T)}")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# colors = ["g", "b", "r", "orange"]
# for cluster, color in zip(clusters, colors):
#     xs = cluster[:, 0]
#     ys = cluster[:, 1]
#     zs = cluster[:, 2]
#     ax.scatter(xs, ys, zs, color=color)

for vec in H:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    #ax.quiver(0, 0, 0, x*mean_len, y*mean_len, z*mean_len, color="r", ls="--")

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# colors = ["g", "b", "r", "orange"]
# for cluster, color in zip(clusters, colors):
#     xs = cluster[:, 0]
#     ys = cluster[:, 1]
#     zs = cluster[:, 2]
#     ax.scatter(xs, ys, zs, color=color)

for vec in H_mv:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    #ax.quiver(0, 0, 0, x*mean_len, y*mean_len, z*mean_len, color="r", ls="--")

plt.show()

