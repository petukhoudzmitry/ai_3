import Sigmoid
import Tanh
import Relu

from enum import Enum

class ActivationFuncs(Enum):
    RELU = 0,
    SIGMOID = 1,
    TANH = 2

__activation_functions = {
    ActivationFuncs.RELU : (Relu.relu, Relu.relu_derivative),
    ActivationFuncs.SIGMOID : (Sigmoid.sigmoid, Sigmoid.sigmoid_derivative),
    ActivationFuncs.TANH : (Tanh.tanh, Tanh.tanh_derivative)
}

def get_activation_function(n):
    return __activation_functions.get(n, __activation_functions[ActivationFuncs.RELU])