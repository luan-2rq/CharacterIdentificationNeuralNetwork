from MLP import *
import pandas as pd
from training_data import TrainingTuple
from utils import *

def main():
    
    #Leitura do arquivo do conjunto de dados
    df_limpo = np.genfromtxt('..\data\caracteres-limpo.csv', delimiter=",", encoding='UTF-8-sig')
    #df_ruido = pd.read_csv(data/caracteres-ruido.csv', delimiter=',', encoding='UTF-8')
    #df_ruido_20 = pd.read_csv(data/caracteres-ruido-20.csv', delimiter=',', encoding='UTF-8')
    # data = df_limpo[:,0:2]
    # labels = df_limpo[:,2:3]
    data = df_limpo[:,0:63]
    labels = df_limpo[:,63:70]

    #Cada caractere tera 63 pixeis que os representam, portanto terao 63 neuronios de entrada
    n_neurons_input = 63
    
    #Este parametro deve ser definido para o nosso problema atraves de experimentacao
    n_hidden_layers_neurons = [10] #1 camada com 15 neuronios
    
    #Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
    n_neurons_output = 7

    #Taxa de aprendizado deve ser entre ]0, 1] (nao pode ser inicializado com 0)
    learning_rate = 0.3

    mlp_limpo = MLP(n_neurons_input, n_neurons_output, n_hidden_layers_neurons, learning_rate)
    # mlp_limpo.test_feed_forward()
    training_data = []
    for i in range(len(data)):
        training_data.append(TrainingTuple(data[i], labels[i]))
    mlp_limpo.train(training_data, 1500)
    
    for i in range(len(training_data)):
        mlp_limpo.predict(training_data[i])


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



