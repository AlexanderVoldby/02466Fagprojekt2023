import numpy as np
import torch.linalg


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
        if self.ploss is None:
            return False
        else:
            return abs(self.ploss - self.loss)/abs(self.ploss) < self.alpha
