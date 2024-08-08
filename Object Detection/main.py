import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# example of one neuron with all of its inputs
# each input should have an associated weight
# X = [[1.0, 2.0, 3.0, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]
#
# X, y = spiral_data(100, 3)

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2.0, 3.0, 0.5]
#
# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]
#
# # np.array().T does a transpose (rotates the matrix on its side)
# # this allows the dot product to be able to happen
#
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs)

# can be as many neurons in the layer as you want (5 in this case)
# n_inputs in next layer has to be n_neurons of layer before
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # ReLU
        # (inputs * weights) + bias
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # backward propagation
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

'''
* Activation Functions
- helps solve problems that involve non-linear equations

* step activation function - in a single neuron, if, weights*inputs + bias, is
  greater than zero, neuron fires a 1. Otherwise, fire 0.

* linear activation function - usually applied to last layers output in the case
  of a regression model.

* sigmoid activation function - Good for when it comes time to optimize weights and
  biases. y = 1 / (1+e^-x). Returns 0 for -inf. 0.5 for zero. 1 for +inf.

* Rectified Linear Units activation function (ReLU) - easier to calc than sigmoid.
  y = { x when x>0
      { 0 when x<=0
'''


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# converts raw output scores into probabilities by taking the exponential of each output
# and then normalizing each value by dividing the sum of all exponentials
class Activation_Softmax:
    def forward(self, outputBatches):
        # subtracting max prevents overflow
        # To each batch, it'll subtract the max to each num
        exp_values = np.exp(outputBatches - np.max(outputBatches, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

'''
* Softmax Loss (categorical cross entropy)
- Used for multiclass classification. Outputs probability over N classes for each image
* Raw Outputs -> Softmax activation = vector of predicted probabilities over input classes
* one hot encoding - converting information ito a format that may be fed into an ML Alg to improve prediction.

ex: 
raw_output = [0.7, 0.1, 0.2]
target_class = 0 (basically tells where you are in the one-hot encoding and what raw_output your looking at)
one_hot = [1, 0, 0]

L = -(math.log(raw_output[0])*one_hot[0] + 
      math.log(raw_output[1])*one_hot[1] + 
      math.log(raw_output[2])*one_hot[2])
  = 0.35667494393873245
  
L simplifies to "-math.log(raw_output[0])" because everything else zero's out
closer to zero = higher confidence
'''

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(selfself, y_pred, y_true):
        samples = len(y_pred)

        # "1e-7" is used to be min value SUPER close to zero because
        # negative infinity problem
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            # grabs certain values from samples that we want to compare
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # if target is given in 2d array (one-hot-encoded)
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # num samples
        samples = len(dvalues)
        # num labels in every sample
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # calc gradient of loss function to predictions
        self.dinputs = -y_true / dvalues_clipped
        # normalize by num of samples
        self.dinputs = self.dinputs / samples

# Stochastic gradient decent optimizer
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_parameters(self, layer):
        # updates weights/biases layers by subtracting learning rate * gradients
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()
optimizer = Optimizer_SGD()

for data in range(10001):
    # forwards the data through each layer of the network
    # forwards raw data to first layer
    dense1.forward(X)
    # activation function on first layer
    activation1.forward(dense1.output)
    # forwards batches of output to 2nd layer
    dense2.forward(activation1.output)
    # softmax function on 2nd layer (output batches)
    activation2.forward(dense2.output)
    # takes in raw output and target
    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    # print loss and accuracy at intervals of a 1000
    if data % 1000 == 0:
        print(f'data: {data}, loss: {loss:.3f}, accuracy: {accuracy:.3f}')

    # Backward pass through loss and layers
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimizer.update_parameters(dense1)
    optimizer.update_parameters(dense2)

