import numpy as np

def forward(X, act_fun, weights, biases):
    activations = []
    z_values = []

    for i in range(len(weights)):
        z = np.dot(X, weights[i]) + biases[i]
        z_values.append(z)
        X = act_fun(z) if i < len(weights) - 1 else z
        activations.append(X)

    return activations, z_values


def backward(X, y, activations, z_values, weights, biases, derivative):
    gradients_w = [None] * len(weights)
    gradients_b = [None] * len(biases)

    y_pred = activations[-1]
    error = y_pred - y
    dz = error / X.shape[0]

    for i in reversed(range(len(weights))):
        gradients_w[i] = np.dot(activations[i - 1].T if i > 0 else X.T, dz)
        gradients_b[i] = np.sum(dz, axis=0)

        if i > 0:
            dz = np.dot(dz, weights[i].T) * derivative(z_values[i - 1])

    return gradients_w, gradients_b
