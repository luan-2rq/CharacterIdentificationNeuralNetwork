import MLP.py

#Cada caractere terah 63 pixeis que os representam, portanto terao 63 neuronios de entrada
n_neurons_input = 63
#Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
n_neurons_output = 7
#Este parametro deve ser definido para o nosso problema
n_hidden_layers = 1

mlp = MLP(n_neurons_input, n_neurons_output, n_hidden_layers)


#data e label sao dois conjuntos de dados que devem ter o mesmo tamanho. Data deve conter vetores de dados e label deve conter 
#a classicacao de cada um desses vetores. Data e label serao o conjunto de dados de treino.

#Para o proposito deste ep, data irah possuir vetores de 63 entradas valoradas com 0 ou 1.
data = []
#Label sera um matriz de largura 7, jah que possuem 7 tipos de caracteres a serem analisados
label = []

MLP.train(data, label)

