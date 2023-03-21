from torchNMFtopy import torchNMF
# from torchAAtopy import torchAA
import numpy as np
import matplotlib.pyplot as plt

# Create artificial 3d dataset of 4 gaussian clusters
m1 = [3., 3., 3.]
m2 = [4., 7., 3.]
m3 = [5., 3., 3.]
m4 = [5., 5., 8.]
sigma = 1
cov = np.eye(3)*sigma

clusters = np.array([np.random.multivariate_normal(m, cov, size=25) for m in [m1, m2, m3, m4]])
# reshape data into 100 x 3 data matrix
X = clusters.reshape((100, 3))

# Run NMF, AA and min-volume NMF and plot

nmf = torchNMF(X.T, 4) # Use X.T to get basis matrix representation of W.
W, H = nmf.run(verbose=True)
# Columns of W now represent basis vectors and row i from X can be rebuilt as sum(Wh_i)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(clusters.shape)

colors = ["g", "b", "r", "orange"]
basis = [col for col in W.T]
for cluster, color in zip(clusters, colors):
    xs = cluster[:, 0]
    ys = cluster[:, 1]
    zs = cluster[:, 2]
    ax.scatter(xs, ys, zs, color=color)

for vec in basis:
    x, y, z = vec
    ax.quiver(0, 0, 0, x, y, z)
    ax.quiver(0, 0, 0, x*10, y*10, z*10, headlength=0, headaxislength=0, ls="--")

plt.show()


