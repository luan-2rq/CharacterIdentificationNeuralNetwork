from cgi import test
import random
import logging

from pyrsistent import l
from sklearn import datasets
from utils import *
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    KFold
)


class MLP(object):
    # n_output_neurons eh uma lista, assim eh possivel determinar o numero de neuronios em cada uma das hidden layers
    def __init__(self, n_input_neurons = 63, n_output_neurons = 7, n_hidden_layers_neurons = [15], learning_rate = 0.2, activation_func = sigmoid_func, activation_func_derivative = sigmoid_derivative_func, output_func = step_func):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.n_hidden_layers_neurons = n_hidden_layers_neurons
        self.learning_rate = learning_rate
        self.weights = [] # weights é um array de matrizes contendo todos os pesos da rede
        self.delta_weights = []
        self.activations = [] # ativacoes dos neuronios
        self.derivatives = [] #derivadas por neuronio
        self.induced_fields = [] #induced fields
        self.local_gradients = [] #local gradients
        self.n_neurons = [] # lista contendo o numero de neuronios de cada camada
        self.epochs = 0

        self.n_neurons.append(self.n_input_neurons)
        self.n_neurons.extend(self.n_hidden_layers_neurons)
        self.n_neurons.append(self.n_output_neurons)

        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative

        for i in range(len(self.n_neurons)):
            self.activations.append(np.zeros((self.n_neurons[i])))
            self.induced_fields.append(np.zeros((self.n_neurons[i])))
            self.local_gradients.append(np.zeros((self.n_neurons[i])))

    # Inicialização dos pesos
    def init_weights(self):  
        np.random.seed(12345)
        for i in range(len(self.n_neurons)-1):
            weight_layer = np.random.uniform(-1,1, (self.n_neurons[i], self.n_neurons[i+1]))
            delta_weights_layer = np.random.uniform(0,0, (self.n_neurons[i], self.n_neurons[i+1]))
            self.weights.append(weight_layer)
            self.delta_weights.append(delta_weights_layer)
        
    def feed_forward(self, input):
        ## Camada de entrada ##

        self.activations[0] = input[:]
        self.induced_fields[0] = input[:]

        ## Camadas Intermediarias ##

        for i, current_weigths in enumerate(self.weights[:-1]):
            self.induced_fields[i+1] = np.dot(self.activations[i], current_weigths)
            self.activations[i+1] = self.activation_func(self.induced_fields[i+1])

        ## Saida ##
        
        last_weights = self.weights[-1]
        self.induced_fields[-1] = np.dot(self.activations[-2], last_weights)
        self.activations[-1] = step_func(self.induced_fields[-1], 0)

        return self.activations[-1]

    def back_propagate(self, expected_output):

        ## Camada de Saida ##

        error = expected_output - self.activations[-1]

        self.local_gradients[-1] = error * self.activation_func_derivative(self.activations[-1])
        
        output_local_gradients = np.array(self.local_gradients[-1], ndmin=2)
        previous_activations = self.activations[-2].reshape(self.activations[-2].shape[0], -1)

        self.delta_weights[-1] = np.dot(previous_activations, output_local_gradients) * self.learning_rate

        ## Camadas Escondidas ## 

        for i in reversed(range(len(self.weights)-1)):
            ## Calculo Gradientes Locais para camadas escondidas ##
            local_gradients_in = np.dot(self.local_gradients[i+2], self.weights[i+1].T)
            self.local_gradients[i+1] = local_gradients_in * self.activation_func_derivative(self.induced_fields[i+1])

            previous_activations = self.activations[i].reshape(self.activations[i].shape[0], -1)

            self.delta_weights[i] = np.dot(previous_activations, np.array(self.local_gradients[i+1], ndmin=2)) * self.learning_rate

    def train(self, dataset_training, dataset_validation, max_epoch, min_accuracy):
        #Variaveis da Validacao
        accuracy = 0
        sum_training_instant_errors = 0
        mean_sqrt_error_training = 0

        sum_test_instant_errors = 0
        mean_sqrt_error_test = 1

        #Passo 0
        self.init_weights()

        ## Salvando Pesos Inciais ## 
        self.save_initial_weights()
        ##########################

        training_data = dataset_training[:, :-self.n_neurons[-1]]
        test_data = dataset_validation[:, :-self.n_neurons[-1]]
        training_labels = dataset_training[:, self.n_neurons[0]:]
        test_labels = dataset_validation[:, self.n_neurons[0]:]

        stop_condition = False
        #Passo 1
        while not stop_condition:
            ## Preparing data

            #Executando Epocas
            print(f"###########################Epoca: {self.epochs+1}#################################")

            sum_training_instant_errors = 0
            #Passos 3, 4 e 5
            for i, data in enumerate(training_data):
                #feed_forward
                input = data
                expected_output = training_labels[i]
                output = self.feed_forward(input)

                instant_error = self.mean_squad_error(output, expected_output)
                sum_training_instant_errors += instant_error

            #Passos 6 e 7 
                #backpropagation
                self.back_propagate(expected_output)
                # self.print_weights()
                # self.print_delta_weights()
            #Passo 8
                #Weights update
                for i in range(len(self.weights)):  
                    self.weights[i] = self.weights[i] + self.delta_weights[i]
                
            mean_sqrt_error_training = sum_training_instant_errors / len(training_data)

            print(f"Erro Quadrado Medio Treinamento: {mean_sqrt_error_training}")

            ######## Teste ########
            sum_test_instant_errors = 0
            correct_predictions_test = 0
            for i, data in enumerate(test_data):

                input = data
                expected_output = test_labels[i]
                output = self.feed_forward(input)

                instant_error = self.mean_squad_error(output, expected_output)
                sum_test_instant_errors += instant_error

                correct_predictions_test += 1 if instant_error == 0 else 0

            mean_sqrt_error_test = sum_test_instant_errors / len(test_data)
            
            print(f"Erro Quadrado Medio Teste: {mean_sqrt_error_test}\n")

            #######################

            ######### Acuracia Teste ###########

            accuracy = (correct_predictions_test/len(test_data))
            print(f'Acuracia Teste {accuracy}')

            #######################

            self.epochs += 1

            #Passo 9(com parada antecipada)
            if self.epochs >= max_epoch or (min_accuracy < accuracy and abs(mean_sqrt_error_training - mean_sqrt_error_test) < 0.17): ## substituir pela real condição para parada
                stop_condition = True
                print(f"Treinamento realizado em {self.epochs} epocas.")   
                print(f"A acuracia final da validaçao foi {accuracy}.")

    # Soma de todos os sqrt errors da camada de saida dividido por 2
    def instant_error(self, output, expected_output) -> float:
        sum_sqrt_error = 0
        for i in range(len(output)):
            sum_sqrt_error += sqrt_error(expected_output[i], output[i])
        instant_error = 0.5*sum_sqrt_error
        return instant_error

    def mean_squad_error(self, label, output):
        return np.average((label - output)**2)

    def predict(self, input):
        outputs = []
        for tuple in input:
            output = self.feed_forward(tuple)
            print(f"Saida 1: {self.answer(output)}\n")
            outputs.append(output)


    def save_initial_weights(self):
        initial_weigths_path = ""
    
    def answer(self, output):
        answer = ""
        letters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        for i in range(len(letters)):
            if output[i] == 1:
                answer += letters[i]
            else:
                answer += '.'
        return answer
        # Exemplos:
        #     A: 1000000
        #     B: 0100000
        #     C: 0010000
        #     D: 0001000
        #     E: 0000100
        #     J: 0000010
        #     K: 0000001

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
        print('\n#####Activations#####')
        for line in self.activations:
            print (*line)
        print('####################')
        print('\n')
        

    def print_induced_fields(self):
        print('\n#####Induced Fields#####')
        for line in self.induced_fields:
            print (*line)
        print('#######################')
        print('\n')

    def print_local_gradients(self):
        print('\n#####Local Gradients#####')
        for line in self.local_gradients:
            print (*line)
        print('########################')
        print('\n')
