import math

def sigmoid_func(x) -> float:
    result = 1/(1+(math.exp(-x)))
    return result

def sigmoid_derivative_func(x) -> float:
    df = sigmoid_func(x)[1-sigmoid_func(x)]
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
