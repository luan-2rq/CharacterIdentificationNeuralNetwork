import math
import numpy as np

def sigmoid_func(x) -> float:
    result = 1.0 / (1.0 + math.exp((-x)))
    return result

def sigmoid_derivative_func(x) -> float:
    df = math.exp(-x) / ((1.0 + math.exp(-x))**2)
    return df

#binary step function
def step_func(x, theta):
    if x >= theta:
        return 1
    else:
        return 0

def error(d, y):
    return d - y

def sqrt_error(d, y):
    return error(d, y)**2

def bipolar(theta, input):
    ativacao = []

    for x in input:
        ativacao.append(
            1 if x >= theta else 0
        )
    return ativacao
