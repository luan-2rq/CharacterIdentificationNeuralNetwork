import random
import logging

from scipy.misc import derivative
from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn.metrics as metrics


class MLP(object):
    # n_output_neurons eh uma lista, assim eh possivel determinar o numero de neuronios em cada uma das hidden layers
    def __init__(self, n_input_neurons = 63, n_output_neurons = 7, n_hidden_layers_neurons = [15], learning_rate = 0.3, bias = True):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.n_hidden_layers_neurons = n_hidden_layers_neurons
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = [] # weights é um array de matrizes contendo todos os pesos da rede
        self.n_neurons = [] # lista contendo o numero de neuronios de cada camada

        layers =  [self.n_input_neurons] + self.n_hidden_layers_neuronsa + [self.n_output_neurons]

        # ativações por camada
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)

        self.activations = activations

        # derivadas por camada
        derivates = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivates.append(d)

        self.derivates = derivates
        
    # Inicialização dos pesos com valores entre [0, 1]
    def init_weights(self):
        self.n_neurons.append(self.n_input_neurons)
        self.n_neurons.extend(self.n_hidden_layers_neurons)
        self.n_neurons.append(self.n_output_neurons)

        # Inicializando os pesos com valores entre 0 e 1(binario, nao bipolar)
        # Entre cada layer existe uma matriz de pesos com index que vao de um neuronio a outro j -> i
        for i in range(len(self.n_neurons)-1):
            n_cols = self.n_neurons[i] + 1 if self.bias else self.n_neurons[i] # +1 por conta do neuronio de bias
            n_rows = self.n_neurons[i+1]
            layer_mat = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]
            self.weights.append(layer_mat)

    def feed_forward(self, input, activation_func):
        if len(input) != self.n_input_neurons:
            error_log = f"The input layer must be {0} in length, but received inputs was {1} in length.".format(self.n_input_neurons, len(input))
            logging.error(error_log)
        
        # Adicionando o bias no input, assim o bias jah eh calculado
        if self.bias:
            input = [1] + input

        nxt_layer_inputs = []
        y_in = 0
        for weigth_layer in self.weigths:
            input = [1] + nxt_layer_inputs if self.bias else nxt_layer_inputs
            nxt_layer_inputs = [0]*len(weigth_layer[0])
            for i in range(len(weigth_layer[0])): # index do neuronio da proxima camada
                for j in range(len(weigth_layer)): # index do neuronio da atual camada
                    y_in += input[j]*weigth_layer[j][i]
                nxt_layer_inputs[i] = activation_func(y_in)
                y_in = 0

        return nxt_layer_inputs
        
    # Calcula o campo induzido de um neuronio; OBS: O campo induzido nao possui activation fucntion aplicada
    def induced_field(self, input, layer, neuron_index, activation_func):
        local_induced_field = 0
        # retorna um erro caso o input não tenha o tamanho correto
        if len(input) != self.n_input_neurons:
            error_log = f"The input layer must be {0} in length, but received inputs was {1} in length.".format(self.n_input_neurons, len(input))
            logging.error(error_log)

        # layer == 0, entao o neuronio pertence a camada de entrada 
        if layer == 0:
            print("Nao sei se deveriamos estar calculando campo induzido para a camada de entrada")
        else:
            next_layer_input = []
            for x in range(layer-1):
                current_weigths = self.weights[x]
                ## Caso o calculo ja esteja nos pesos que se conectam diretamente ao neuronio que se quer calcular o CI, nao eh necessario multiplicar os pesos para todos os neuronios da proxima camada, somente para o neuronio do campo local induzido
                if x == layer-2:
                    for j in range(len(current_weigths)):
                        local_induced_field += self.weights[j][neuron_index]
                else: 
                    for i in range(len(current_weigths[neuron_index])): # index do neuronio da proxima camada
                        for j in range(len(current_weigths)): # index do neuronio da atual camada
                            y_in += self.weigths[j][i] * input[j]
                        next_layer_input[i] = activation_func(y_in)
                    input = next_layer_input
        return local_induced_field

    def weigth_change(self, input, layer, neuron_index, activation_func, activation_func_derivative, expected_output):
        
        weigth_change = 0

        induced_field = self.induced_field(input, layer, neuron_index, activation_func)
        y_j = activation_func(induced_field)
        d_j = expected_output[neuron_index]
        error = error(d_j, y_j)

        delta = error * activation_func_derivative(induced_field)

        if layer == self.weigths.count:
            ##neuronio da camada de saida, portanto passo 6
            if neuron_index == 0:
                ##Neuronio eh bias

                weigth_change = self.learning_rate * delta
            else:
                
                weigth_change = self.learning_rate*delta*y_j
        else:
            ##neuronio de camada escondida, portanto passo 7
            pass
            
        return weigth_change

    def back_propagate(self, input, layer, neuron_index, activation_func, activation_func_derivative, expected_output):
        # calcular novos pesos para caso uma predição tenha sido errada
        weigth_change = 0

        induced_field = self.induced_field(input, layer, neuron_index, activation_func)
        y_j = activation_func(induced_field)
        d_j = expected_output[neuron_index]
        error = error(d_j, y_j)

        delta = error * activation_func_derivative(induced_field)

        if layer == self.weigths.count:
            ##neuronio da camada de saida, portanto passo 6
            if neuron_index == 0:
                ##Neuronio eh bias

                weigth_change = self.learning_rate * delta
            else:
                
                weigth_change = self.learning_rate*delta*y_j
        else:
            ##neuronio de camada escondida, portanto passo 7
            pass
        
        """"
        for i in reversed(range(len(self.derivates))):
            activations = self.activations[i+1]
            delta = error * sigmoid_derivative_func(activations)
            delta_t = delta.reshape(delta.shape[0], -1).T
            current_activation = self.activations[i]
            current_activation = current_activation.reshape(current_activation.shape[0], -1) # transforma em uma matriz coluna
            self.derivates[i] = np.dot(current_activation, delta_t)
            error = np.dot(delta, self.weights[i].T)
        """

    def train(self, features, target):
        self.init_weights()
        ## Save Initial Weigths# # 

        stop_condition = True
        while stop_condition:
            for i in range():
                self.feed_propagate()

        # metodo responsavel pelo treinamento da rede e deve receber um conjunto de dados juntamente com os rotulos de cada vetor de dado
        pass

    # Soma de todos os sqrt errors da camada de saida dividido por 2
    def instant_error(self, output, expected_output) -> float:
        sum_sqrt_error = 0
        for i in range(len(output)):
            sum_sqrt_error += sqrt_error(expected_output[i], output[i])
        instant_error = 0.5*sum_sqrt_error
        return instant_error

    # Soma de todos os erros instantaneos dividido por n(numero de epocas)
    # AKA funcao de custo, uma metrica para calcular o desempenho
    def mean_sqrt_error(self, output) -> float:
        sum_sqrt_error = 0
        for i in range(len(output)):
            sum_sqrt_error += sqrt_error(expected_output[i], output[i])
        error = 0.5*sum_sqrt_error

        #return 

    #def mean_squad_error(self, label, output):
       # return np.average((label - output)**2) lari vai fazer aqui

    def neuron_induced_field(self) -> float:
        pass

    def features_lable(self, dataset):
        X = dataset[:, :-self.n_outputs] #features
        y = dataset[: , self.n_inputs:] #label

        return X, y

    def predict(self, answer):
        # imprime o output das respostas 
        # A: 1000000
        # B: 0100000
        # C: 0010000
        # D: 0001000
        # E: 0000100
        # J: 0000010
        # K: 0000001

        if answer[0] == 1:
            print("A")
        if answer[1] == 1:
            print("B")
        if answer[2] == 1:
            print("C")
        if answer[3] == 1:
            print("D")
        if answer[4] == 1:
            print("E")
        if answer[5] == 1:
            print("J")
        if answer[6] == 1:
            print("K")

