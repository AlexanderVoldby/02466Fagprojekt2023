import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from helpers.data import X

plt.rcParams["figure.figsize"] = (27,3)

print(X.shape)

start, end = 76500, 77000

for i in X:
    plt.plot(i[start:end])
plt.show()
print(np.min(X))

# rec = pd.read_csv('recons_x.csv', header=None, index_col=False)
rec = pd.read_parquet('recons_x_nmf.parquet', engine='fastparquet')
rec = rec.transpose()
rec_X = rec.to_numpy()

# plt.plot(rec_X[1])
print(rec_X.shape)
for i in rec_X:
    plt.plot(i[start:end])

# plt.plot(rec_X[2][start:end]-max(rec_X[2]))
plt.show()