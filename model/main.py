from MLP import *

def main():
    #Cada caractere terah 63 pixeis que os representam, portanto terao 63 neuronios de entrada
    n_neurons_input = 1
        #Este parametro deve ser definido para o nosso problema atraves de experimentacao
    hidden_layers = [3]
    #Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
    n_neurons_output = 1

    #Taxa de aprendizado deve ser entre [0, 1], porem nao pode ser inicializado com 0
    learning_rate = 0.3

    mlp = MLP(n_neurons_input, n_neurons_output, hidden_layers, learning_rate)
    mlp.init_weights()
    #data e label sao dois conjuntos de dados que devem ter o mesmo tamanho. Data deve conter vetores de dados e label deve conter 
    #a classicacao de cada um desses vetores. Data e label serao o conjunto de dados de treino.

    #Para o proposito deste ep, data irah possuir vetores de 63 entradas valoradas com 0 ou 1.
    #data = []
    #Label sera um matriz de largura 7, jah que possuem 7 tipos de caracteres a serem analisados
    #label = []

    #mlp.train(data, label)


if __name__ == "__main__":
    main()






