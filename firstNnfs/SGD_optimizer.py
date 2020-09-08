import numpy as np

class SGD:
	def __init__(self, learningRate = 1, decay = 0.1, momentum = 0):
		self.learningRate = learningRate
		self.decay = decay
		self.iterations = 0
		self.currentLearningRate = learningRate
		self.momentum = momentum

	def preUpdateParameters(self):
		if self.decay:
			self.currentLearningRate = self.currentLearningRate * (1. / (1. + self.decay * self.iterations))




	def updateParameters(self, layer):    #adjust weights and biases by subtaracing learning rate


		# If layer does not contain momentum arrays, create ones 
        #filled with zeros
		if not hasattr(layer, 'weight_momentums'):
			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.bias_momentums = np.zeros_like(layer.biases)

        # If we use momentum
		if self.momentum:

            # Build weight updates with momentum - take previous 
            # updates multiplied by retain factor and update with 
            # current gradients
			weight_updates = ((self.momentum * layer.weight_momentums) - (self.currentLearningRate * layer.dweights))
			layer.weight_momentums = weight_updates

            # Build bias updates
			bias_updates = ((self.momentum * layer.bias_momentums) - (self.currentLearningRate * layer.dbiases))
			layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
		else:
			weight_updates = (-self.currentLearningRate * layer.dweights)
			bias_updates = (-self.currentLearningRate * layer.dbiases)

		layer.weights += weight_updates
		layer.biases += bias_updates
	def postUpdateParameters(self):   #increments iterations
		self.iterations += 1
