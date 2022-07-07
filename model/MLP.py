import random
import logging

from pyrsistent import l
from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold
)


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
            self.induced_fields.append([float('-inf') for _ in range(self.n_neurons[i])])
            self.local_gradients.append([float('-inf') for _ in range(self.n_neurons[i])])
        
    # Inicialização dos pesos com valores entre [0, 1]
    def init_weights(self):
        # Inicializando os pesos com valores entre 0 e 1(binario, nao bipolar)
        # Entre cada layer existe uma matriz de pesos com index que vao de um neuronio a outro j -> i
        # self.weights.append([[0.5, 0.5],
        #         [0.5, 0.5]])
        # self.weights.append([[0.5, 0.5]])
        for i in range(len(self.n_neurons)-1):
            n_cols = self.n_neurons[i] + 1 if self.bias else self.n_neurons[i] # +1 por conta do neuronio de bias
            n_rows = self.n_neurons[i+1]
            weight_layer = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]
            delta_weights_layer = [[0 for _ in range(n_cols)] for _ in range(n_rows)]
            self.weights.append(weight_layer)
            self.delta_weights.append(delta_weights_layer)
        self.print_weights()
        

    def feed_forward(self, input):
        output = []
        self.activations[0] = input[:]
        self.induced_fields[0] = input[:]
        for i in range(len(self.weights)):
            for j in range(self.n_neurons[i+1]):
                layer = i + 1
                neuron_index = j
                # print(f"Induced Fields antes l{layer}: {self.induced_fields[layer]}")
                self.induced_fields[layer][neuron_index] = self.induced_field(self.activations[i], layer, neuron_index)
                self.activations[layer][neuron_index] = self.activation_func(self.induced_fields[layer][neuron_index])
                # print(f"Induced Fields depois l{layer}: {self.induced_fields[layer]}")
                # print(f"Activations depois l{layer}: {self.activations[layer]}")
                # print(f'Induced field: layer{layer} neuron{neuron_index} - {self.induced_fields[layer][neuron_index]}')
                # print(f'Activation: layer{layer} neuron{neuron_index} - {self.activations[layer][neuron_index]}')
            print(f"Local induced field l{i+1}: {self.induced_fields[i+1]}")
        output = bipolar(0, self.induced_fields[len(self.induced_fields)-1])
        # print(self.induced_fields)
        # print(output)
        self.activations[len(self.activations)-1] = output
        return output
        
    # Calcula o campo induzido de um neuronio; OBS: O campo induzido nao possui activation fucntion aplicada
    def induced_field(self, input, layer, neuron_index):
        if self.bias:
            input = np.insert(input, 0, 1)
            
        local_induced_field = 0
        
        # layer == 0, entao o neuronio pertence a camada de entrada 
        if layer == 0:
            print("Nao sei se deveriamos estar calculando campo induzido para a camada de entrada")
        else:
            # for x in range(layer):
            current_weigths = self.weights[layer-1]
            #     ## Caso o calculo ja esteja nos pesos que se conectam diretamente ao neuronio que se quer calcular o CI, nao eh necessario multiplicar os pesos para todos os neuronios da proxima camada, somente para o neuronio do campo local induzido
            for j in range(len(current_weigths[neuron_index])):
                # if neuron_index == 0 and layer == 2:
                #     print(f"The input is {input[j]} and weight {current_weigths[neuron_index][j]}")
                local_induced_field = local_induced_field + (current_weigths[neuron_index][j] * input[j])
                if 1 == layer:
                    print(f"Partial induced field: {(current_weigths[neuron_index][j] * input[j])}")
        return local_induced_field

    def weight_change(self, neuron_layer, local_gradient, weight_i_index):
        weight_change = 0

        if weight_i_index == 0 and self.bias:
            weight_change = self.learning_rate * local_gradient
        else:
            neuron_index = 0
            if self.bias:
                neuron_index = weight_i_index - 1
            weight_change = self.learning_rate * local_gradient * self.activations[neuron_layer-1][neuron_index]
        return weight_change

    def back_propagate(self, expected_output):
        for i in reversed(range(len(self.n_neurons))): #index das camadas
            for j in range(self.n_neurons[i]): #index de cada neuronio da camada atual
                local_gradient = self.local_gradient(i, j, expected_output) # local gradient do neuronio atual
                for k in range(self.n_neurons[i-1]+1 if self.bias else self.n_neurons[i-1]): # k - index da camada anterior 
                    if i > 0: # i > 0, pois delta_weights < n_layers
                        # print(f"Local gradient l{i}:{j} - {local_gradient}")
                        self.delta_weights[i-1][j][k] = self.weight_change(i, local_gradient, k)
                        #print(self.delta_weights[i-1][j][k])
            print(f"Ativacoes l{i}: {self.activations[i]}")
            if i != 0:
                print(f"Local gradients l{i}: {self.local_gradients[i]}")

    def local_gradient(self, layer, neuron_index, expected_output):
        local_gradient = 0
        # print(f"layer ind_fields lenth {len(self.induced_fields[layer])}")
        # print(f"neuron index {neuron_index}")

        derivative = self.activation_func_derivative(self.activations[layer][neuron_index])
        #Caso o neuronio seja da camada de saida
        #Este if esta calculando corretamente o gradiente local
        if layer == len(self.n_neurons)-1:
            error = expected_output[neuron_index] - self.activations[layer][neuron_index]
            local_gradient = error * derivative
            self.local_gradients[layer][neuron_index] = local_gradient
        #Caso o neuronio seja da camada escondida
        #O gradiente local nao esta correto nesse else
        else:
            weights = self.weights[layer]
            for j in range(self.n_neurons[layer+1]):
                local_gradient += self.local_gradients[layer+1][j] * weights[j][neuron_index+1 if self.bias else neuron_index]
            local_gradient = local_gradient * derivative
            self.local_gradients[layer][neuron_index] = local_gradient
        #print(f"Local gradient l{layer}:{neuron_index} - {local_gradient}")
        
        return local_gradient

    def train(self, training_data, maxEpocs):
        #Step 0
        self.init_weights()
        ## Save Initial Weigths ## 

        ##########################

        stop_condition = False

        #step 1
        while not stop_condition:
            # X = training_data[:, :-self.n_neurons[-1]]
            # y = training_data[:, self.n_neurons[0]:]
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.33)

            #Executar epocas
            for train_data in training_data:
            #step 3, 4 e 5 
                #feed_forward
                #pegar entrada
                input = train_data.data
                expected_output = train_data.label
                # print("-------------------")
                # print(f'label {expected_output}')
                output = self.feed_forward(input)
                print(f"Entrada: {input}")
                print(f"Saida: {output}")
                print(f'Expected: {expected_output}\n')
                # print("\n")
                # self.print_weights()
                # print("\n")
                # print("-------------------")

            #step 6 e 7 
                #backpropagation
                self.back_propagate(expected_output)
                self.print_weights()
                self.print_delta_weights()
                #self.print_delta_weights()
            #step 8
                #atualizaçao de pesos
                for i in range(len(self.weights)):  
                # iterate through columns
                    for j in range(len(self.weights[i])):
                        for k in range(len(self.weights[i][j])):
                            self.weights[i][j][k] += self.delta_weights[i][j][k] 

                for i in range(len(self.n_neurons)):
                    self.activations[i] = [float('-inf') for _ in range(self.n_neurons[i])]
                    self.induced_fields[i] = [float('-inf') for _ in range(self.n_neurons[i])]
                    self.local_gradients[i] = [float('-inf') for _ in range(self.n_neurons[i])]
            
            self.epocas += 1

            #step 9
            if self.epocas >= maxEpocs: ## substituir pela real condição para parada
                stop_condition = True
                self.print_weights()

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

    def print_weights(self):
        print('#####WEIGHTS#####')
        for matrix in self.weights:
            for line in matrix:
                print (*line)
            print('\n')
        print('#################')
        print('\n')
    
    def print_delta_weights(self):
        print('\n#####d_weigths#####')
        for matrix in self.delta_weights:
            for line in matrix:
                print (*line)
            print('\n')
        print('#################')
        print('\n')

    def print_activations(self):
        for line in self.activations:
            print (*line)
        print('\n')

    def print_induced_fields(self):
        for line in self.induced_fields:
            print (*line)
        print('\n')

    def print_local_gradients(self):
        print('Local gradients:')
        for line in self.local_gradients:
            print (*line)
        print('\n')

    def predict(self, data_tuple):
        output = self.feed_forward(data_tuple.data)
        print(f"Saida: {output}\n")
        self.print_answer(output)

    def test_feed_forward(self):
        self.init_weights()

        # print(f"Activation result: {self.activation_func(0.5)}")

        self.weights[0][0][1] = 0.5
        self.weights[0][1][1] = 0.5
        self.weights[0][0][2] = 0.5
        self.weights[0][1][2] = 0.5

        #bias
        self.weights[0][0][0] = 0.5
        self.weights[0][1][0] = 0.5

        #bias
        self.weights[1][0][0] = 0.5

        self.weights[1][0][1] = 0.5
        self.weights[1][0][2] = 0.2

        output = self.feed_forward([1, -1])
        for i in range(1):
            self.back_propagate([0])

        #self.print_local_gradients()

        self.print_delta_weights()

        # for i in range(len(self.weights)):  
        #     # iterate through columns
        #         for j in range(len(self.weights[i])):
        #             for k in range(len(self.weights[i][j])):
        #                 self.weights[i][j][k] += self.delta_weights[i][j][k] 

        # self.print_weights()

        # output = self.feed_forward(np.array([1, -1]))

        # print()

        # self.print_weights()

        # self.print_activations()
        # print('Expected 1.12')
        # print(output)

    def print_answer(self, output):
       # imprime o output das respostas
        if output[0] == 1:
            print("True")
        else:
            print("False")
        # print('Resposta: \n')
        # answer = ""
        # letters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        # for i in range(len(letters)):
        #     if output[i] == 1:
        #         answer += letters[i]
        #     else:
        #         answer += '.'
        # return answer
        # Exemplos:
        #     A: 1000000
        #     B: 0100000
        #     C: 0010000
        #     D: 0001000
        #     E: 0000100
        #     J: 0000010
        #     K: 0000001

