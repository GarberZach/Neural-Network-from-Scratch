import numpy as np

class Loss:

    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = y_pred.shape[0]

        

        if len(y_true.shape) == 1:
            y_pred = y_pred[range(samples), y_true]

        # Losses
        negative_log_likelihoods = -np.log(y_pred)

        if len(y_true.shape) == 2: #gets rid of 
            negative_log_likelihoods *= y_true

        # Average loss
        data_loss = np.sum(negative_log_likelihoods) / samples
        return data_loss

    def backward(self, dvalues, y_true):

        samples = dvalues.shape[0]

        self.dvalues = dvalues.copy()  # Copy to safely modify
        self.dvalues[range(samples), y_true] -= 1
        self.dvalues = self.dvalues / samples
