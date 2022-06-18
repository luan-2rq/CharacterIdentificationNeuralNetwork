import random
import logging

from numpy import array
from utils import *

class MLP(object):
    #n_output_neurons eh uma lista, 
    #assim eh possivel determinar o numero de
    #neuronios em cada uma das hidden layers
    def __init__(self, n_input_neurons = 63, n_output_neurons = 7, hidden_layers = [15], learning_rate = 0.3, bias = True):
        self.n_input_neurons = n_input_neurons
        self.n_output_neurons = n_output_neurons
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.bias = bias
        #weights é uma matriz contendo todos os pesos, o primeiro index é referente ao neuronio a qual ele sai e o segundo index é referente ao neuronio que ele se conecta
        self.weights = []
        
    #Inicialização dos pesos com valores entre [0, 1]
    def init_weights(self):
        #lista contendo o numero de neuronios de cada camada
        n_neurons = []
        n_neurons.append(self.n_input_neurons)
        n_neurons.extend(self.hidden_layers)
        n_neurons.append(self.n_output_neurons)
        print(n_neurons)
        #inicializando os pesos com valores entre 0 e 1(binario, nao bipolar)
        #Entre cada layer existe uma matriz de pesos com index que vao de um neuronio a outro i -> j

        for i in range(len(n_neurons)-1):
            n_cols = n_neurons[i] + 1 if self.bias else n_neurons[i] # +1 por conta do neuronio de bias
            n_rows = n_neurons[i+1]
            layer_mat = [[random.random() for _ in range(n_cols)] for _ in range(n_rows)]
            self.weights.append(layer_mat)

    def feed_forward(self, input, activation_func) -> array:
        if len(input) != self.n_input_neurons:
            error_log = f"The input layer must be {0} in length, but received inputs was {1} in length.".format(self.n_input_neurons, len(input))
            logging.error(error_log)
        
        #Adicionando o bias no input, assim o bias jah eh calculado sem validacao posterior
        if self.bias:
            input = [1] + input

        nxt_layer_inputs = []
        sum = 0
        for weigth_layer in self.weigths:
            input = [1] + nxt_layer_inputs if self.bias else nxt_layer_inputs
            nxt_layer_inputs = [0]*len(weigth_layer[0])
            for i in range(len(weigth_layer[0])): # index do neuronio da proxima camada
                for j in range(len(weigth_layer)): # index do neuronio da primeira camada
                    sum += input[j]*weigth_layer[j][i]
                nxt_layer_inputs[i] = activation_func(sum)
                sum = 0

        return nxt_layer_inputs
        

    def back_propagate(self):
        #calcular novos pesos para caso uma predição tenha sido errada
        pass

    def train(self, data, label):
        #este metodo sera responsavel pelo treinamento da rede e deve receber um conjunto de dados juntamente com os rotulos de cada vetor de dado
        pass

    #sum of the sqrt errors of all the output neurons error diveded by two
    def instant_error(self) -> float:
        pass

    #sum of all the instant errors divided by n
    #AKA cost function, a metric to calculate the learning performance
    def mean_sqrt_error(self) -> float:
        pass