import numpy as np

class AdaGrad:    #adaptive gradient changes gradient per feature rather than a flat change for all parameters
	def __init__(self, learningRate = 1, decay = 0.1, momentum = 0, epsilon = 1e-7 ):
		self.learningRate = learningRate
		self.decay = decay
		self.iterations = 0
		self.currentLearningRate = learningRate
		self.momentum = momentum
		self.epsilon = epsilon 

	def preUpdateParameters(self):
		if self.decay:
			self.currentLearningRate = self.currentLearningRate * (1. / (1. + self.decay * self.iterations))

	def updateParameters(self, layer):    #adjust weights and biases by subtaracing learning rate

		if not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)

		weight_updates = (-self.currentLearningRate *                      
							layer.dweights)
		bias_updates = (-self.currentLearningRate * 
							layer.dbiases)

		#update caches with squared gradients
		layer.weight_cache += layer.dweights ** 2
		layer.bias_cache += layer.dbiases ** 2

		layer.weights += -self.currentLearningRate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.currentLearningRate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

	def postUpdateParameters(self):   #increments iterations
		self.iterations += 1
