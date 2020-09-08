import numpy as np

class RmsProp:

    # Initialize optimizer - set settings
    def __init__(self, learningRate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learningRate = learningRate
        self.currentLearningRate = learningRate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # Call once before any parameter updates
    def preUpdateParameters(self):
        if self.decay:
            self.currentLearningRate = self.currentLearningRate * (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def updateParameters(self, layer):

        # If layer does not contain cache arrays,
        # create ones filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.currentLearningRate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.currentLearningRate *layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after any parameter updates
    def postUpdateParameters(self):
        self.iterations += 1
