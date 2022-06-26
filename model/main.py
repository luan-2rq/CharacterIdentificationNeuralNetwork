from MLP import *
import pandas as pd

def main():
    
    #Leitura do arquivo do conjunto de dados
    df_limpo = pd.read_csv(data/caracteres-limpo.csv', delimiter=',', encoding='UTF-8')
    #df_ruido = pd.read_csv(data/caracteres-ruido.csv', delimiter=',', encoding='UTF-8')
    #df_ruido_20 = pd.read_csv(data/caracteres-ruido-20.csv', delimiter=',', encoding='UTF-8')

    #Cada caractere tera 63 pixeis que os representam, portanto terao 63 neuronios de entrada
    n_neurons_input = 63
    
    #Este parametro deve ser definido para o nosso problema atraves de experimentacao
    n_hidden_layers_neurons = [15] #1 camada com 15 neuronios
    
    #Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
    n_neurons_output = 7

    #Taxa de aprendizado deve ser entre ]0, 1] (nao pode ser inicializado com 0)
    learning_rate = 0.3

    mlp_limpo = MLP(n_neurons_input, n_neurons_output, n_hidden_layers_neurons, learning_rate)
    features, label = mlp_limpo.features_lable(df_limpo)
    mlp_limpo.train(features, label)
    mlp_limpo.predict()


    print("\n---------------- Resultados ----------------")
    print("Acurácia: {}".format((metrics.accuracy_score(target, testes))))
    print("Precisão: {}".format(metrics.precision_score(target, testes, average='micro')))
    print("Recall: {}".format(metrics.recall_score(target, testes, average='micro')))
    print("F1_score: {}".format(metrics.f1_score(target, testes, average='micro')))
    print("Roc_Auc_score: {}".format(
        metrics.roc_auc_score(target, testes, average='micro')))
    
    m_c = metrics.confusion_matrix(target.argmax(axis=1), testes.argmax(axis=1))
    print(m_c)

    df_cm = pd.DataFrame(m_c, index=[i for i in "ABCDEJK"],
                        columns=[i for i in "ABCDEJK"])
    plt.figure(figsize=(7, 6))
    sn.heatmap(df_cm, annot=True)
    plt.show()




