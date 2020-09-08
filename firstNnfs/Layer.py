import random
import numpy as np
from Relo import Relo
from softmax import Softmax
from loss import Loss
from SGD_optimizer import SGD
from AdaGrad_optimizer import AdaGrad
from RmsProp_optimizer import RmsProp
from Adam_optimizer import Adam
random.seed(0)
np.random.seed(0)



def create_data(n, k):
    X = np.zeros((n*k, 2))  # data matrix (each row = single example)
    y = np.zeros(n*k, dtype='uint8')  # class labels
    for j in range(k):
        ix = range(n*j, n*(j+1))
        r = np.linspace(0.0, 1, n)  # radius
        t = np.linspace(j*4, (j+1)*4, n) + np.random.randn(n)*0.2  # theta
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = j
    return X, y

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def passForward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)	

X, y = create_data(100, 3)

layer1 = Layer(2, 60)
layer2 = Layer(60, 3)

activation1 = Relo()
activation2 = Softmax()
lossFunc = Loss()
#optimizer = SGD(decay = 5e-8, momentum = 0.7)
#optimizer = AdaGrad(decay = 1e-8)
#optimizer = RmsProp(decay = 1e-8)
optimizer = Adam(learningRate = 0.05, decay = 4e-8)

for epoch in range(10001):

	layer1.passForward(X)

	activation1.forward(layer1.output)

	layer2.passForward(activation1.output)

	activation2.forward(layer2.output)

	

	#print(lossFunc.forward(activation2.output, y))
	loss = lossFunc.forward(activation2.output, y)

	# Calculate accuracy from output of activation2 and targets
	predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
	accuracy = np.mean(predictions==y)

	

	#-------------------------------------------------------------------------------------------------------
	#back propagation

	lossFunc.backward(activation2.output, y)
	activation2.backward(lossFunc.dvalues)
	layer2.backward(activation2.dvalues)
	activation1.backward(layer2.dvalues)
	layer1.backward(activation1.dvalues)

	if not epoch % 100:
	        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.currentLearningRate}')

	optimizer.preUpdateParameters()
	optimizer.updateParameters(layer1)
	optimizer.updateParameters(layer2)
	optimizer.postUpdateParameters()

#Create test data
X_test, y_test = create_data(100, 3)


	#repeat process with test data
layer1.passForward(X_test)

activation1.forward(layer1.output)

layer2.passForward(activation1.output)

activation2.forward(layer2.output)

	

	#print(lossFunc.forward(activation2.output, y))
loss = lossFunc.forward(activation2.output, y_test)

	# Calculate accuracy from output of activation2 and targets
predictions = np.argmax(activation2.output, axis=1)  # calculate values along first axis
accuracy = np.mean(predictions==y_test)


print(f' Validation data: acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {optimizer.currentLearningRate}')