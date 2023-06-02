from torchNMF import NMF, MVR_NMF
# from torchAAtopy import torchAA
from helpers.callbacks import explained_variance
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import torch
# Create artificial 3d dataset of 4 gaussian clusters
np.random.seed(5)
m = np.array([20, 20, 20])
sigma = 20
cov = np.eye(3)*sigma
cluster = np.random.multivariate_normal(m, cov, size=75)
# reshape data into 100 x 3 data matrix
X = cluster.reshape((75, 3))
# Run NMF, AA and min-volume NMF and plot

nmf = NMF(X, 3)
W, H = nmf.fit()
print(f"Volume of H: {np.linalg.det(H@H.T)}")
print(f"Explained variance using NMF: {explained_variance(X, nmf.forward().detach().numpy())}")

nmf_min_vol = MVR_NMF(X, 3)
W_mv, H_mv = nmf_min_vol.fit()
print(f"Volume of H_mv: {np.linalg.det(H_mv@H_mv.T)}")
print(f"Explained variance using MVR NMF: {explained_variance(X, nmf_min_vol.forward().detach().numpy())}")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = cluster[:, 0]
ys = cluster[:, 1]
zs = cluster[:, 2]
ax.scatter(xs, ys, zs)

# Calculate normal vector of the tetrahedron and its vertices
H = H*10
normal = np.cross(H[1]-H[0], H[2]-H[0])
vertices = np.array([H[0], H[1], H[2], H[0] + normal])
vectors = np.vstack((H, [0, 0, 0]))
convex_hull = ConvexHull(vectors)
base_indices = convex_hull.vertices[:-1]
base_vertices = vertices[base_indices]
base_vertices = np.vstack((base_vertices, np.array([0, 0, 0])))
for vec in H:
    x, y, z = vec
    ax.quiver(0, 0, 0, x*2, y*2, z*2, color="r", ls="--", arrow_length_ratio=0)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    ax.plot_trisurf(base_vertices[:, 0], base_vertices[:, 1], base_vertices[:, 2],
                    triangles=[[0, 1, 2], [0, 2, 3], [0, 3, 1]],
                    cmap='viridis', alpha=0.1)

plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = cluster[:, 0]
ys = cluster[:, 1]
zs = cluster[:, 2]
ax.scatter(xs, ys, zs)

H_mv = H_mv*10
normal = np.cross(H_mv[1]-H_mv[0], H_mv[2]-H_mv[0])
vertices = np.array([H_mv[0], H_mv[1], H_mv[2], H_mv[0] + normal])
vectors = np.vstack((H_mv, [0, 0, 0]))
convex_hull = ConvexHull(vectors)
base_indices = convex_hull.vertices[:-1]
base_vertices = vertices[base_indices]
base_vertices = np.vstack((base_vertices, np.array([0, 0, 0])))
for vec in H_mv:
    x, y, z = vec
    ax.quiver(0, 0, 0, x*2, y*2, z*2, color="r", ls="--", arrow_length_ratio=0)
    # Scale NMF basis with mean cluster length, otherwise the vectors are too short for visualization
    ax.plot_trisurf(base_vertices[:, 0], base_vertices[:, 1], base_vertices[:, 2],
                    triangles=[[0, 1, 2], [0, 2, 3], [0, 3, 1]],
                    cmap='viridis', alpha=0.1)

plt.show()

