from MLP import *
import pandas as pd
from utils import *
from math import *
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sn

def main():

    #Carregando dados 
    df_limpo = np.array(pd.read_csv(r"..\data\caracteres-limpo.csv", delimiter=",", header=None, encoding='UTF-8'))
    df_ruido = np.array(pd.read_csv(r"..\data\caracteres-ruido.csv", delimiter=",", header=None, encoding='UTF-8'))
    df_ruido_20 = pd.read_csv(r"..\data\caracteres_ruido20.csv", delimiter=",", header=None, encoding='UTF-8')

    #Conjunto de treino
    training_dataset = np.array(df_limpo)

    #Conjunto de validacao
    validation_dataset = np.array(df_ruido)

    ##Conjunto que servira de teste
    test_dataset = np.array(df_ruido_20)

    #Cada caractere tera 63 pixeis que os representam, portanto terao 63 neuronios de entrada
    n_neurons_input = 63
    #Camadas escondida, contendo 5 camadas 
    n_hidden_layers_neurons = [30]
    #Serao 7 classes diferentes de caracteres, portanto serao 7 neuronios de saida
    n_neurons_output = 7
    #Taxa de aprendizado deve ser entre [0, 1] (nao pode ser inicializado com 0)
    learning_rate = 0.0012

    #Inicializando a instancia da rede neural
    mlp = MLP(n_neurons_input, n_neurons_output, n_hidden_layers_neurons, learning_rate)

    #Treinando a rede
    mlp.train(training_dataset, validation_dataset, 2000, early_stopping=True, min_accuracy=0.7, min_mean_sqrt_error_training=0.015)

    #Conjunto de teste
    test_dataset_test_data = test_dataset[:,0:63]
    test_dataset_labels = test_dataset[:,63:70]

    print("###########################Teste - Resultados#################################")

    #Predicoes feitas pela rede a partir do conjunto de teste. Eh devolvido um array com as saidas pelo metodo predict.
    predictions = mlp.predict(test_dataset_test_data)

    print(f"Acuracia: {metrics.accuracy_score(test_dataset_labels, predictions)}")
    print(f"Precisao: {metrics.precision_score(test_dataset_labels, predictions, average='micro')}")
    print(f"Recall: {metrics.recall_score(test_dataset_labels, predictions, average='micro')}")
    print(f"F1 Score: {metrics.f1_score(test_dataset_labels, predictions, average='micro')}")
    print(f"ROC AUC Score: {metrics.roc_auc_score(test_dataset_labels, predictions, average='micro')}")
    
    confusion_matrix = metrics.confusion_matrix(np.argmax(test_dataset_labels, axis=1), np.argmax(predictions, axis=1))

    confusion_matrix_data = pd.DataFrame(confusion_matrix, index=[i for i in "ABCDEJK"],
                        columns=[i for i in "ABCDEJK"])
    plt.figure(figsize=(7, 6))
    sn.heatmap(confusion_matrix_data, annot=True)
    plt.show()

if __name__ == "__main__":
    main()



