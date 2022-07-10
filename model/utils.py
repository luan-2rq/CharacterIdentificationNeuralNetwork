from math import *
import numpy as np

##Retorna um array contendo a aplicacao da funcao para cada item do array de entrada recebida
def sigmoid_func(input):
    activations = []

    for x in input:
        activation = 1.0 / (1.0 + exp((-x)))
        activations.append(activation)
    return np.array(activations)

##Retorna um array contendo a aplicacao da funcao para cada item do array de entrada recebida
def sigmoid_derivative_func(input):
    derivatives = []

    for x in input:
        derivative = exp(-x) / ((1.0 + exp(-x))**2)
        derivatives.append(derivative)
    return np.array(derivatives)

##Retorna um array contendo a aplicacao da funcao para cada item do array de entrada recebida
def step_func(input, theta):
    result = []
    for x in input:
        if x >= theta:
            result.append(1)
        else:
            result.append(0)
    return np.array(result)

def error(d, y):
    return d - y

def sqrt_error(d, y):
    return error(d, y)**2
