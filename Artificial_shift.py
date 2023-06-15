import numpy as np

#Create data
# Define random sources, mixings and shifts; H, W and tau
N, M, d = 30, 10000, 3
Fs = 1000  # The sampling frequency we use for the simulation
t0 = 10    # The half-time interval we look at
t = np.arange(-t0, t0, 1/Fs)  # the time samples
f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

def gauss(mu, s, time):
    return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)


def shift_dataset(W, H, tau):
    # Get half the frequencies
    Nf = H.shape[1] // 2 + 1
    # Fourier transform of S along the second dimension
    Hf = np.fft.fft(H, axis=1)
    # Keep only the first Nf[1] elements of the Fourier transform of S
    Hf = Hf[:, :Nf]
    # Construct the shifted Fourier transform of S
    Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
    # Concatenate the original columns with the reversed columns along the second dimension
    Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
    f = np.arange(0, M) / M
    omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
    Wf = np.einsum('Nd,NdM->NdM', W, omega)
    # Broadcast Wf and H together
    Vf = np.einsum('NdM,dM->NM', Wf, Hft)
    V = np.fft.ifft(Vf)
    return V

np.random.seed(42)

# Random mixings:
W = np.random.dirichlet(np.ones(d), N)
W = np.append(W, [[1,0,0]], axis=0)
W = np.append(W, [[0,1,0]], axis=0)
W = np.append(W, [[0,0,1]], axis=0)
N = N+3

#W = np.random.rand(N, d)
shift = 200
# Random gaussian shifts
tau = np.random.randint(-shift, shift, size=(N, d))
#tau = np.random.randint(0, 1000, size=(N, d))
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
mean = [40, 300, 700]
std = [10, 20, 7]
t = np.arange(0, 1000, 0.1)
H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

X = shift_dataset(W, H, tau)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchAA import torchAA
    from torchNMF import NMF
    plt.figure()
    for signal in H:
        plt.plot(signal)
    plt.title("Original signals")
    plt.show()
    plt.figure()
    for signal in X:
        plt.plot(signal.real)
    plt.title("Dataset build from mixing of the three sources")
    plt.show()

    plt.figure()
    plt.imshow(W, aspect='auto', interpolation="none")
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(np.arange(0, d, 1))
    ax.set_yticks(np.arange(0, N, 1))
    plt.title("W - The mixings")
    plt.show()