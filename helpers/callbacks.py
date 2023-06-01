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


class EarlyStop(Stopper):
    def __init__(self, patience = 5, offset = 0) -> None:
        self.patience = patience
        
        self.counter = 0
        self.lowest = np.Inf
        
        self.offset = offset
        
    def track_loss(self, loss_val):
        if loss_val < self.lowest + self.offset:
            self.lowest = loss_val
            self.counter = 0
        else:
            self.counter += 1
            
    def trigger(self):
        return self.counter > self.patience


class RelativeStopper:

    def __init__(self, data, alpha=1e-6):
        self.norm = torch.linalg.matrix_norm(data, ord="fro").item()**2
        self.alpha = alpha
        self.loss = 1e9

    def track_loss(self, loss):
        self.loss = loss

    def trigger(self):
        return self.loss/self.norm < self.alpha


# 
class ChangeStopper:
    def __init__(self, alpha=1e-8):
        self.alpha = alpha
        self.ploss = None
        self.loss = None

    def track_loss(self, loss):
        if self.loss is None:
            self.loss = loss

        else:
            self.ploss = self.loss
            self.loss = loss

    def trigger(self):
        if self.ploss is None:
            return False
        else:
            return abs((self.ploss - self.loss)/self.ploss) < self.alpha
