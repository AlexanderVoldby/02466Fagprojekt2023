import numpy as np
import matplotlib.pyplot as plt
from helpers.data import X

# Get a sample from X:
print(f"Matrix has shape {X.shape}")
start, end = 76500, 77000
xf = X[0]
plt.plot(xf[start:end])
plt.show()
# Data is in frequency domain, so we apply the inverse Fourier transform.
xt = np.fft.ifft(xf)
plt.plot(xt.real, label="real")
plt.plot(xt.imag, "--", label="Imaginary")
plt.legend()
plt.show()