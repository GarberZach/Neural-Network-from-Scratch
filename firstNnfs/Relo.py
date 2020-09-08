import numpy as np
class Relo:

	def forward(self, inputs):

		self.inputs = inputs

		self.output = np.maximum(0, inputs)           
		#self.output = 1.0 / (1.0 + np.exp(-inputs))  #sigmoid

	def backward(self, dvalues):
        #copy dvalues so weights and biases can safely be changed 
		self.dvalues = dvalues.copy()

        # if the rectified linear changes output to zero
        #then obviously the gradient is too so this just skips calculations
		self.dvalues[self.inputs <= 0] = 0 
