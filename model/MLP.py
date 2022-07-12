import sys
from utils import *
import numpy as np
import os

class MLP(object):
    # n_output_neurons eh uma lista, assim eh possivel determinar o numero de neuronios em cada uma das hidden layers
    def __init__(self, n_input_neurons = 63, n_output_neurons = 7, n_hidden_layers_neurons = [15], learning_rate = 0.2, activation_func = sigmoid_func, activation_func_derivative = sigmoid_derivative_func, output_func = output_func):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.n_hidden_layers_neurons = n_hidden_layers_neurons
        self.learning_rate = learning_rate
        self.weights = [] #Pesos da rede
        self.delta_weights = [] #Alteracao de pesos, eh alterado durante o backpropagation
        self.activations = [] #Ativacoes dos neuronios
        self.induced_fields = [] #Campos Locais Induzidos
        self.local_gradients = [] #Gradientes Locais
        self.n_neurons = [] # lista contendo o numero de neuronios de cada camada
        self.epochs = 0

        self.n_neurons.append(self.n_input_neurons)
        self.n_neurons.extend(self.n_hidden_layers_neurons)
        self.n_neurons.append(self.n_output_neurons)

        self.activation_func = activation_func
        self.activation_func_derivative = activation_func_derivative
        self.output_func = output_func

        for i in range(len(self.n_neurons)):
            self.activations.append(np.zeros((self.n_neurons[i]+1)))
            self.induced_fields.append(np.zeros((self.n_neurons[i]+1)))
            self.local_gradients.append(np.zeros((self.n_neurons[i])))

    # Inicialização dos pesos
    def init_weights(self):  
        np.random.seed(25475)
        for i in range(len(self.n_neurons)-1):
            weight_layer = np.random.uniform(-1,1, (self.n_neurons[i]+1, self.n_neurons[i+1]))
            delta_weights_layer = np.random.uniform(0,0, (self.n_neurons[i]+1, self.n_neurons[i+1]))
            self.weights.append(weight_layer)
            self.delta_weights.append(delta_weights_layer)
        
    def feed_forward(self, input):
        ## Camada de entrada ##

        input = np.insert(input, 0, 1)
        self.activations[0] = input[:]
        self.induced_fields[0] = input[:]

        ## Camadas Intermediarias ##

        for i, current_weigths in enumerate(self.weights[:-1]):
            induced_field = np.dot(self.activations[i][1:], current_weigths[1:,:])

            ## Configurando o Bias ##
            induced_field = np.insert(induced_field, 0, 1)
            #########################

            self.induced_fields[i+1] = induced_field
            self.activations[i+1] = self.activation_func(self.induced_fields[i+1])

            ## Configurando o Bias ##
            self.activations[i+1][0] = 1
            #########################

        ## Camada de Saida ##
        
        last_weights = self.weights[-1][1:,:]
        induced_field = np.dot(self.activations[-2][1:], last_weights)
        induced_field = np.insert(induced_field, 0, 1)

        self.induced_fields[-1] = induced_field
        self.activations[-1] = self.activation_func(self.induced_fields[-1])

        return self.output_func(self.activations[-1][1:])

    def back_propagate(self, expected_output):

        ## Camada de Saida ##

        error = expected_output - self.activations[-1][1:]

        self.local_gradients[-1] = error * self.activation_func_derivative(self.activations[-1][1:])
        
        output_local_gradients = np.array(self.local_gradients[-1], ndmin=2)
        previous_activations = self.activations[-2].reshape(self.activations[-2].shape[0], -1)

        self.delta_weights[-1] = np.dot(previous_activations, output_local_gradients) * self.learning_rate

        ## Camadas Escondidas ## 

        for i in reversed(range(len(self.weights)-1)):
            ## Calculo Gradientes Locais para camadas escondidas ##
            local_gradients_in = np.dot(self.local_gradients[i+2], self.weights[i+1][1:,:].T)
            self.local_gradients[i+1] = local_gradients_in * self.activation_func_derivative(self.induced_fields[i+1][1:])

            previous_activations = self.activations[i].reshape(self.activations[i].shape[0], -1)

            self.delta_weights[i] = np.dot(previous_activations, np.array(self.local_gradients[i+1], ndmin=2)) * self.learning_rate

    def train(self, dataset_training, dataset_validation, max_epoch, early_stopping, min_accuracy, min_mean_sqrt_error_training):
        #Variaveis da Validacao
        validation_accuracy = 0
        sum_training_instant_errors = 0
        mean_sqrt_error_training = 0

        sum_test_instant_errors = 0
        current_mean_sqrt_error_test = 0
        previous_mean_sqrt_error_test = 1

        #Passo 0
        self.init_weights()

        ## Salvando Pesos Inciais ## 
        self.save_weights("initial_weights.csv")
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

                instant_error = self.instant_error(output, expected_output)
                sum_training_instant_errors += instant_error

            #Passos 6 e 7 
                #backpropagation
                self.back_propagate(expected_output)

            #Passo 8
                #Atualizacao de pesos
                for i in range(len(self.weights)):  
                    self.weights[i] = self.weights[i] + self.delta_weights[i]
                
            mean_sqrt_error_training = sum_training_instant_errors / len(training_data)

            print(f"Erro Quadratico Medio Treinamento: {mean_sqrt_error_training}")

            if early_stopping:
                ######## Validacao ########
                sum_test_instant_errors = 0
                correct_predictions_test = 0
                for i, data in enumerate(test_data):

                    input = data
                    expected_output = test_labels[i]
                    output = self.feed_forward(input)

                    instant_error = self.instant_error(output, expected_output)
                    sum_test_instant_errors += instant_error

                    correct_predictions_test += 1 if instant_error == 0 else 0

                previous_mean_sqrt_error_test = current_mean_sqrt_error_test
                current_mean_sqrt_error_test = sum_test_instant_errors / len(test_data)
                
                print(f"Erro Quadratico Medio Validacao: {current_mean_sqrt_error_test}")

                #######################

                ######### Validacao Acuracia ###########

                validation_accuracy = (correct_predictions_test/len(test_data))
                print(f'Acuracia Validacao: {validation_accuracy}\n')

                #######################

            self.epochs += 1

            #Passo 9(com parada antecipada)
            if self.epochs >= max_epoch:
                stop_condition = True
                self.save_weights("final_weights_no_early_stopping.csv")
                print(f"Treinamento realizado em {self.epochs} epocas.")   
            elif early_stopping:
                if (min_accuracy <= validation_accuracy and previous_mean_sqrt_error_test <= current_mean_sqrt_error_test and mean_sqrt_error_training < min_mean_sqrt_error_training):
                    stop_condition = True
                    self.save_weights("final_weights_early_stopping.csv")
                    print(f"Treinamento realizado em {self.epochs} epocas.")   
                    print(f"A acuracia final da validaçao foi {validation_accuracy}.")
                    


    # Soma de todos os sqrt errors da camada de saida dividido por 2
    def instant_error(self, output, expected_output) -> float:
        sum_sqrt_error = 0
        for i in range(len(output)):
            sum_sqrt_error += sqrt_error(expected_output[i], output[i])
        instant_error = 0.5*sum_sqrt_error
        return instant_error

    def predict(self, dataset, dataset_labels):
        outputs = []
        for i, tuple in enumerate(dataset):
            output = self.feed_forward(tuple)
            print(f"Saida {i+1}: {self.answer(output)}   Saida Esperada: {self.answer(dataset_labels[i])}")
            outputs.append(output)
        return outputs

    def answer(self, output):
        answer = ""
        letters = ['A', 'B', 'C', 'D', 'E', 'J', 'K']
        for i in range(len(letters)):
            if output[i] == 1:
                answer += letters[i]
            else:
                answer += '.'
        return answer
        # Exemplos de answer:
        #     A: A......
        #     B: .B.....
        #     C: ..C....
        #     D: ...D...
        #     E: ....E..
        #     J: .....J.
        #     K: ......K
    
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

    def save_weights(self, file_name):
        np.set_printoptions(threshold=sys.maxsize)
        try:
            folder = r"..\useful_files"
            os.mkdir(folder)
        except:
            pass
        initial_weigths_path = r"..\useful_files\{}".format(file_name)
        np.savetxt(initial_weigths_path, self.weights, delimiter=',', fmt='%s')
    
