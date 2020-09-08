import numpy as np

class Softmax:

	def forward(self, inputs):

		self.inputs = inputs

		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # subtracting keeps values from 
		#becoming too large as subtracting a constant from all values will change nothing in the terms of outputed
		#normalized values

		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		#axis, and keepdims are paremeters to make sure this funnction returns an output that is the same as the input 
		#in terms of array dimensions

		self.output = probabilities

	def backward(self, dvalues):
		self.dvalues = dvalues.copy() #copy gradients values so other values can be safely adjusted and retain the 
    	#originals 