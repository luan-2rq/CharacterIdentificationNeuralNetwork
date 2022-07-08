from MLP import *
import pandas as pd
from training_data import TrainingTuple
from utils import *
from math import *

def main():
    
    #Leitura do arquivo do conjunto de dados
    df_limpo = pd.read_csv(r"C:\Users\Luan Monteiro\MLP_Neural_Network-Implementation\data\caracteres-limpo.csv", delimiter=",", header=None, encoding='UTF-8-sig')
    df_ruido = pd.read_csv(r"C:\Users\Luan Monteiro\MLP_Neural_Network-Implementation\data\caracteres-ruido.csv", delimiter=",", header=None, encoding='UTF-8')
    df_ruido_20 = pd.read_csv(r"C:\Users\Luan Monteiro\MLP_Neural_Network-Implementation\data\caracteres_ruido20.csv", delimiter=",", header=None, encoding='UTF-8')
    # data = df_limpo[:,0:2]
    # labels = df_limpo[:,2:3]
    # data = df_limpo[:,0:63]
    # labels = df_limpo[:,63:70]

    dataset = np.concatenate((df_limpo, df_ruido, df_ruido_20))
    # dataset = np.array(df_limpo)

    #Cada caractere tera 63 pixeis que os representam, portanto terao 63 neuronios de entrada
    n_neurons_input = 63
    
    #Este parametro deve ser definido para o nosso problema atraves de experimentacao
    n_hidden_layers_neurons = [49, 39, 33, 29] #1 camada com 15 neuronios
    
    #Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
    n_neurons_output = 7

    #Taxa de aprendizado deve ser entre ]0, 1] (nao pode ser inicializado com 0)
    learning_rate = 0.02

    mlp_limpo = MLP(n_neurons_input, n_neurons_output, n_hidden_layers_neurons, learning_rate)
    # mlp_limpo.test_feed_forward()

    training_data_percentage = 0.7
    data_size = len(dataset)
    training_data_end = floor(training_data_percentage * data_size)

    training_dataset = dataset[0:training_data_end,:]
    test_dataset = dataset[training_data_end:data_size,:]

    mlp_limpo.train(training_dataset, test_dataset, 20000, 0.7)
    
    for i in range(len(dataset)):
        mlp_limpo.predict(dataset[i])


    # # print("\n---------------- Resultados ----------------")
    # # print("Acurácia: {}".format((metrics.accuracy_score(target, testes))))
    # # print("Precisão: {}".format(metrics.precision_score(target, testes, average='micro')))
    # # print("Recall: {}".format(metrics.recall_score(target, testes, average='micro')))
    # # print("F1_score: {}".format(metrics.f1_score(target, testes, average='micro')))
    # # print("Roc_Auc_score: {}".format(
    # #     metrics.roc_auc_score(target, testes, average='micro')))
    
    # # m_c = metrics.confusion_matrix(target.argmax(axis=1), testes.argmax(axis=1))
    # # print(m_c)

    # # df_cm = pd.DataFrame(m_c, index=[i for i in "ABCDEJK"],
    # #                     columns=[i for i in "ABCDEJK"])
    # # plt.figure(figsize=(7, 6))
    # # sn.heatmap(df_cm, annot=True)
    # # plt.show()

if __name__ == "__main__":
    main()



