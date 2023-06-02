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
            return abs(self.ploss - self.loss)/abs(self.ploss) < self.alpha
