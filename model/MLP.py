import random
import logging

from matplotlib.font_manager import _Weight
from utils import *
import numpy as np
import pandas as pd


class MLP(object):
    # n_output_neurons eh uma lista, assim eh possivel determinar o numero de neuronios em cada uma das hidden layers
    def __init__(self, n_input_neurons = 63, n_output_neurons = 7, n_hidden_layers_neurons = [15], learning_rate = 0.3, bias = True, activation_func = sigmoid_func, activation_func_derivative = sigmoid_derivative_func):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.n_hidden_layers_neurons = n_hidden_layers_neurons
        self.learning_rate = learning_rate
        self.bias = bias
        self.weights = [] # weights é um array de matrizes contendo todos os pesos da rede
        self.delta_weights = []
        self.activations = [] # ativacoes dos neuronios
        self.derivatives = [] #derivadas por neuronio
        self.induced_fields = [] #induced fields
        self.local_gradients = [] #local gradients
        self.n_neurons = [] # lista contendo o numero de neuronios de cada camada
        self.epocas = 0

        self.n_neurons.append(self.n_input_neurons)
        self.n_neurons.extend(self.n_hidden_layers_neurons)
        self.n_neurons.append(self.n_output_neurons)

        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative

        for i in range(len(self.n_neurons)):
            self.activations.append([float('-inf') for _ in range(self.n_neurons[i])])

        for i in range(len(self.n_neurons)):
            self.derivatives.append([float('-inf') for _ in range(self.n_neurons[i])])

        for i in range(len(self.n_neurons)):
            self.induced_fields.append([float('-inf') for _ in range(self.n_neurons[i])])

        for i in range(len(self.n_neurons)):
            self.local_gradients.append([float('-inf') for _ in range(self.n_neurons[i])])
        
    # Inicialização dos pesos com valores entre [0, 1]
    def init_weights(self):
        # Inicializando os pesos com valores entre 0 e 1(binario, nao bipolar)
        # Entre cada layer existe uma matriz de pesos com index que vao de um neuronio a outro j -> i
        for i in range(len(self.n_neurons)-1):
            n_cols = self.n_neurons[i] + 1 if self.bias else self.n_neurons[i] # +1 por conta do neuronio de bias
            n_rows = self.n_neurons[i+1]
            weight_layer = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]
            delta_weights_layer = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
            self.weights.append(weight_layer)
            self.delta_weights.append(delta_weights_layer)
        print(self.weights)

    def feed_forward(self, input):
        output = []
        for i in range(len(self.weights)):
            layer_inputs = input[:]
            for j in range(self.n_neurons[i+1]):
                    layer = i + 1
                    neuron_index = j
                    self.induced_fields[layer][neuron_index] = self.induced_field(layer_inputs, layer, neuron_index)
                    self.activations[layer][neuron_index] = self.activation_func(self.induced_fields[layer][neuron_index])
            layer_inputs = self.activations[layer][:]
        output = layer_inputs
        return output
        
    # Calcula o campo induzido de um neuronio; OBS: O campo induzido nao possui activation fucntion aplicada
    def induced_field(self, input, layer, neuron_index):
        local_induced_field = 0
        # retorna um erro caso o input não tenha o tamanho correto
        if len(input) != self.n_input_neurons:
            error_log = f"The input layer must be {0} in length, but received inputs was {1} in length.".format(self.n_input_neurons, len(input))
            logging.error(error_log)

        # layer == 0, entao o neuronio pertence a camada de entrada 
        if layer == 0:
            print("Nao sei se deveriamos estar calculando campo induzido para a camada de entrada")
        else:
            for x in range(layer):
                current_weigths = self.weights[x]
                ## Caso o calculo ja esteja nos pesos que se conectam diretamente ao neuronio que se quer calcular o CI, nao eh necessario multiplicar os pesos para todos os neuronios da proxima camada, somente para o neuronio do campo local induzido
                if x == layer-1:
                    for j in range(len(current_weigths)):
                        local_induced_field += current_weigths[neuron_index][j] * input[j]
                else: 
                    for i in range(len(current_weigths[neuron_index])): # index do neuronio da proxima camada
                        for j in range(len(current_weigths)): # index do neuronio da atual camada
                            y_in += current_weigths[i][j] * input[j]
                        input[i] = self.activation_func(y_in)
        return local_induced_field

    def weight_change(self, neuron_layer, local_gradient, weight_i_index):
        weight_change = 0
        if weight_i_index == 0:
            weight_change = self.learning_rate * local_gradient
        else:
            weight_change = self.learning_rate * local_gradient * self.activations[neuron_layer-1][weight_i_index]
        return weight_change

    def back_propagate(self, output, expected_output):
        for i in reversed(range(len(self.n_neurons))):
            for j in range(self.n_neurons[i]):
                for k in range(self.n_neurons[i-1]):
                    if i > 0:
                        local_gradient = self.local_gradient(i, j, expected_output)
                        self.delta_weights[i-1][j][k] = self.weight_change(i, local_gradient, k)

    def local_gradient(self, layer, neuron_index, expected_output):
        local_gradient = 0
        # print(f"layer {layer}")
        # print(f"neuron index {neuron_index}")
        derivative = self.activation_func_derivative(self.induced_fields[layer-1][neuron_index])
        #Caso o neuronio seja da camada de saida
        if layer == len(self.n_neurons)-1:
            error = expected_output[neuron_index] - self.activations[layer][neuron_index]
            local_gradient = error * derivative
        #Caso o neuronio seja da camada escondida
        else:
            weights = self.weights[layer]
            for i in range(self.n_neurons[layer]):
                for j in range(self.n_neurons[layer+1]):
                    if self.local_gradients[layer][j] != float('-inf'):
                        local_gradient += local_gradient[layer][j] * weights[j][neuron_index]   
                    else:
                        self.local_gradient[layer][j] = self.local_gradient(layer, j, self.activation_func_derivative)
                        local_gradient += self.local_gradient[layer][j] * weights[j][neuron_index]
            local_gradient = local_gradient * derivative
        self.local_gradients[layer-1][neuron_index] = local_gradient
        return local_gradient

    def train(self, training_data):
        #Step 0
        self.init_weights()
        ## Save Initial Weigths ## 

        ##########################

        stop_condition = False

        #step 1
        while not stop_condition:

            #Executar epoca
            for train_data in training_data:
            #step 3, 4 e 5 
                #feed_forward
                #pegar entrada    
                input = train_data.data
                expected_output = train_data.label
                output = self.feed_forward(input)
            #step 6 e 7 
                #backpropagation
                self.back_propagate(output, expected_output)
            #step 8
                #atualizaçao de pesos
                for i in range(len(self.weights)):  
                # iterate through columns
                    for j in range(len(self.weights[i])):
                        for k in range(len(self.weights[i][j])):
                            self.weights[i][j][k] += self.delta_weights[i][j][k] 

                for i in range(len(self.n_neurons)):
                    self.activations.append([float('-inf') for _ in range(self.n_neurons[i])])

                for i in range(len(self.n_neurons) - 1):
                    self.induced_field.append([float('-inf') for _ in range(self.n_neurons[i])])

                for i in range(len(self.n_neurons) - 1):
                    self.local_gradients.append([float('-inf') for _ in range(self.n_neurons[i])])
            #step 9
            if self.epoca > 1000: ## substituir pela real condição para parada
                stop_condition = True
                print(self.weights)
    

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
    def mean_sqrt_error(self, output, expected_output) -> float:
        sum_sqrt_error = 0
        for i in range(len(output)):
            sum_sqrt_error += sqrt_error(expected_output[i], output[i])
        mean_sqrt_error = 0.5*sum_sqrt_error

        return mean_sqrt_error

    def print_answer(self, output):
        # imprime o output das respostas
        answer = ""
        letters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        for i in range(len(letters)):
            if output[i] == 1:
                answer.append(letters[i])
            else:
                answer.append('.')
        return answer
        #Exemplos:
            # A: 1000000
            # B: 0100000
            # C: 0010000
            # D: 0001000
            # E: 0000100
            # J: 0000010
            # K: 0000001

