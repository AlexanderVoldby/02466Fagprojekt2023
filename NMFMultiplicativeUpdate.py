import numpy as np
import matplotlib.pyplot as plt
from helpers.data import X
from helpers.callbacks import earlyStop

class MultiNMF:
    def __init__(self, x, d):
        # Initialize positive matrices
        self.X = x
        print(self.X.shape)
        self.W = np.random.rand(self.X.shape[0], d)
        self.H = np.random.rand(d, self.X.shape[1])
        print(self.W.shape)
        print(self.H.shape)

    def updateW(self):
        """
        Multiplicative update which reduces euclidean distance ||X - WH||
        """
        VHt = self.X @ np.transpose(self.H)
        WHHt = self.W @ self.H @ np.transpose(self.H)
        self.W = self.W * (VHt / WHHt)

    def updateH(self):
        """
        Multiplicative update which reduces euclidean distance ||X - WH||
        """
        WtX = np.transpose(self.W) @ self.X
        WtWH = np.transpose(self.W) @ self.W @ self.H
        self.H = self.H * (WtX / WtWH)

    def EuclidLoss(self):
        return np.linalg.norm(self.X - self.W @ self.H)

# Define training function for NMF using euclidean distance
def multiplicativeNMF(matrix, d):
    nmf = MultiNMF(matrix, d)
    es = earlyStop()
    while not es.trigger():
        nmf.updateH()
        nmf.updateW()
        loss = nmf.EuclidLoss()
        print(f"Euclidean distance: {loss}")

        es.count(loss)


if __name__ == "__main__":
    multiplicativeNMF(X, 10) # 10 components