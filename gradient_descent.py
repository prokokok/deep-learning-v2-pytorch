import numpy as np

# Activation (sigmoid) function
def sigmoid(x):
    x = np.float_(x)
    return 1 / (1 + np.exp(-x))


# Output (prediction) formula
def output_formula(features, weights, bias):
    #     print(features.shape)
    #     print(weights.shape)
    x = np.dot(features, weights) + bias
    return sigmoid(x)


# Error (log-loss) formula
def error_formula(y, output):
    y = np.float_(y)
    p = np.float_(output)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    w_update = weights + learnrate * (y - output_formula(x, weights, bias)) * x
    b_update = bias + learnrate * (y - output_formula(x, weights, bias))
    return w_update, b_update