from ActivationFuncs import *

from BackPropagation import forward, backward
from Optimizers import Optimizers
from Util import *


class Network:
    def __init__(self, l_rate, ep, b_size, tol, verbose=False, decay_rate=0.99, hidden_layers=[64, 32], activation_func=ActivationFuncs.RELU,
                 early_stopping_patience=10, momentum=0.9, optimizer=Optimizers.SGD, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.optimizer = optimizer

        # Common fields
        self.learning_rate = l_rate
        self.epochs = ep
        self.hidden_layers = hidden_layers
        self.batch_size = b_size
        self.tolerance = tol
        self.verbose = verbose
        self.decay_rate = decay_rate
        self.activation_func = get_activation_function(activation_func)
        self.early_stopping_patience = early_stopping_patience

        # Common fields
        self.weights = []
        self.biases = []

        # Momentum specific fields
        self.momentum = momentum
        self.velocities_w = []
        self.velocities_b = []

        # ADAM specific fields
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.moment_w = []  # First moment (mean of gradients) for weights
        self.moment_b = []  # First moment for bias
        self.t = 0  # Time step for bias correction


    def initialize(self, input_size, output_size):
        # Common part
        layer_sizes = [input_size] + self.hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.random.randn(1, layer_sizes[i + 1]))

            if self.optimizer == Optimizers.SGD_MOMENTUM:
                self.velocities_w.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
                self.velocities_b.append(np.zeros((layer_sizes[i + 1])))
            elif self.optimizer == Optimizers.ADAM:
                self.velocities_w.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
                self.velocities_b.append(np.zeros((1, layer_sizes[i + 1])))
                self.moment_w.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
                self.moment_b.append(np.zeros((1, layer_sizes[i + 1])))


    def predict(self, X):
        activations, _ = forward(X, self.activation_func[0], self.weights, self.biases)
        return activations[-1]


    def fit(self, X, y, X_test, y_test):
        n_samples, n_features = X.shape
        output_size = y.shape[1] if len(y.shape) > 1 else 1
        self.initialize(n_features, output_size)

        training_loss = []
        testing_loss = []

        last_loss = np.inf
        no_improvement_count = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                activations, z_values = forward(X_batch, self.activation_func[0], self.weights, self.biases)
                gradients_w, gradients_b = backward(X_batch, y_batch, activations, z_values, self.weights, self.biases, self.activation_func[1])

                for j in range(len(self.weights)):
                    if self.optimizer == Optimizers.SGD_MOMENTUM:
                        self.velocities_w[j] = self.momentum * self.velocities_w[j] + self.learning_rate * \
                                               gradients_w[j]
                        self.velocities_b[j] = self.momentum * self.velocities_b[j] + self.learning_rate * \
                                               gradients_b[j]

                        # Update weights and bias
                        self.weights[j] -= self.velocities_w[j]
                        self.biases[j] -= self.velocities_b[j]

                    elif self.optimizer == Optimizers.ADAM:
                        self.moment_w[j] = self.beta1 * self.moment_w[j] + (1 - self.beta1) * gradients_w[j]
                        self.moment_b[j] = self.beta1 * self.moment_b[j] + (1 - self.beta1) * gradients_b[j]

                        # Update second moment estimates (variance of gradients)
                        self.velocities_w[j] = self.beta2 * self.velocities_w[j] + (1 - self.beta2) * (
                                    gradients_w[j] ** 2)
                        self.velocities_b[j] = self.beta2 * self.velocities_b[j] + (1 - self.beta2) * (
                                    gradients_b[j] ** 2)

                        # Bias correction
                        m_w_hat = self.moment_w[j] / (1 - self.beta1 ** (self.t + 1))
                        m_b_hat = self.moment_b[j] / (1 - self.beta1 ** (self.t + 1))
                        v_w_hat = self.velocities_w[j] / (1 - self.beta2 ** (self.t + 1))
                        v_b_hat = self.velocities_b[j] / (1 - self.beta2 ** (self.t + 1))

                        # Update weights and bias
                        self.weights[j] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                        self.biases[j] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                    else:
                        self.weights[j] -= self.learning_rate * gradients_w[j]
                        self.biases[j] -= self.learning_rate * gradients_b[j]

            y_pred = self.predict(X)
            loss = mean_squared_error(y, y_pred)

            if self.verbose:
                training_loss.append(loss)

                y_pred = self.predict(X_test)
                loss = mean_squared_error(y_test, y_pred)
                testing_loss.append(loss)

                if epoch % 100 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss}')

            if loss < last_loss:
                last_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.early_stopping_patience:
                if self.verbose:
                    print('Early stopping triggered')
                break

            if np.linalg.norm(gradients_w[-1]) < self.tolerance:
                if self.verbose:
                    print('Convergence reached')
                break

            self.learning_rate *= self.decay_rate

            self.t += 1

        if self.verbose:
            plot_error(min(self.epochs, len(training_loss)), training_loss, testing_loss)

        return self.weights, self.biases