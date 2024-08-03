# when you get a batch of outputs that look something
# like: [-9, -4, 1, 5, 6)
# softmax activation takes the output and converts the negatives
# to a positives - makes everything add up to one.
# the more positive the value, the more its significance is

import numpy as np
import nnfs

nnfs.init()
# batch of outputs
layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

# step - Exponentiate
exp_values = np.exp(layer_outputs)

# "axis=1" sums rows of 2d matrix. "axis=0" would do columns
# "keepdims=True" keeps same format
# print(np.sum(layer_outputs, axis=1, keepdims=True))

# step - Normalize
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)
