import matplotlib.pyplot as plt

from MSE import *

def compute_determination(y_true, y_predict):
    mean = np.mean(y_true, axis=0)
    return 1 - (np.sum((y_true - y_predict) ** 2) / np.sum((y_true - mean) ** 2))


def compute_error_mean(y_true, y_predict):
    return abs(y_true - y_predict).mean()


def plot_error(epochs, training_errors, testing_errors):
    plt.figure(figsize=(20, 12))
    plt.plot(range(epochs), training_errors, label="Training Error", color="blue")
    plt.plot(range(epochs), testing_errors, label="Testing Error", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Training vs. Testing Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def test_model(model, X, y, y_max, y_min, title):
    y_pred = model.predict(X)

    loss = mean_squared_error(y, y_pred)

    y_pred = y_pred * (y_max - y_min) + y_min
    y = y * (y_max - y_min) + y_min

    R = compute_determination(y, y_pred)
    error = compute_error_mean(y, y_pred)

    print(f'{title}\nLoss: {loss}\nCoefficient of determination: %f\nMean difference: {error}' % R)

    for i in range(len(y_pred)):
        print(f"Predicted cost: %d, actual cost: %d, difference: %d" % (
        y_pred[i][0], y[i][0], abs(y_pred[i][0] - y[i][0])))


def test_binary_model(model, X, y, title):
    y_pred = model.predict(X)
    loss = mean_squared_error(y, y_pred)
    R = compute_determination(y, y_pred)

    print(f'{title}\nLoss: {loss}\nCoefficient of determination: {R}')

    for i in range(len(y_pred)):
        print(f'Predicted value: {y_pred[i][0]} = {int(np.round(y_pred[i][0]))}, actual value: {y[i][0]}')