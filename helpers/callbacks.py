import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_data(X, title=""):
    plt.figure()
    for signal in X:
        plt.plot(signal)
    plt.title(title)
    plt.show()


def explained_variance(original_data, reconstructed_data):
    """
    Calculate the explained variance between original and reconstructed data.

    Args:
        original_data (numpy.ndarray): The original dataset.
        reconstructed_data (numpy.ndarray): The reconstructed dataset.

    Returns:
        float: The explained variance score.
    """
    numerator = np.sum(np.square(original_data - reconstructed_data))
    denominator = np.sum(np.square(original_data - np.mean(original_data)))
    explained_variance = 1 - (numerator / denominator)

    return explained_variance

#Superclass for stopping criteria
class Stopper:
    def __init__(self) -> None:
        pass

    # Function for tracking loss - to be implemented in subclasses
    def track_loss(self):
        pass
    
    # Function for triggering stop - to be implemented in subclasses
    def trigger(self):
        pass
    
    # Function for resetting stopper - to be implemented in subclasses
    def reset(self):
        pass

class ChangeStopper(Stopper):
    def __init__(self, alpha=1e-8, patience=5):
        self.alpha = alpha
        self.ploss = None
        self.loss = None
        
        self.patience = patience
        self.counter = 0

    def track_loss(self, loss):
        if self.loss is None:
            self.loss = loss

        else:
            self.ploss = self.loss
            self.loss = loss
        
        if self.ploss is not None:
            if abs((self.ploss - self.loss)/self.ploss) < self.alpha:
                self.counter += 1
            else:
                self.counter = 0

    def trigger(self):
        return self.counter >= self.patience

    def reset(self):
        self.ploss = None
        self.loss = None
        self.counter = None

class ConvergenceCriterion:
    def __init__(self, x, convergence_threshold, num_epochs_convergence):
        self.norm = torch.linalg.matrix_norm(x, ord="fro")**2
        self.convergence_threshold = convergence_threshold
        self.num_epochs_convergence = num_epochs_convergence
        self.previous_loss = float('inf')
        self.epochs_since_last_improvement = 0

    def trigger(self, current_loss):
        current_loss /= self.norm
        self.previous_loss /= self.norm
        if torch.abs(self.previous_loss - current_loss)/self.previous_loss < self.convergence_threshold:
            self.epochs_since_last_improvement += 1
        else:
            self.epochs_since_last_improvement = 0
        print(f"Loss difference {torch.abs(self.previous_loss - current_loss)/self.previous_loss}")
        self.previous_loss = current_loss

        return self.epochs_since_last_improvement >= self.num_epochs_convergence