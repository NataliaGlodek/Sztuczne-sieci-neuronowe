import numpy as np
import pandas as pd
import xlsxwriter

import Network as N
import analiza

# IMPORT DANYCH
data_final = analiza.Wczytywanie_danych()

#stworzenie tabei z x i y
y_data = data_final['Survived']
x_data = data_final.drop('Survived', axis = 1)

#modyfikacja y na hot coding
y_set_hot_coding = []
for i in y_data.tolist():
    if i == 1:
        y_set_hot_coding.append([1, 0])

    elif i == 0:
        y_set_hot_coding.append([0, 1])

#podział na zbior uczący i testowy
y_train_set = y_set_hot_coding[1:800]
x_train_set = x_data[1:800]

y_test_set =  y_set_hot_coding[801:889]
x_test_set =  x_data[801:889]


for j in range(10):
    liczba_neuronow = 2
    # tworze pierwsza warstwe
    layer1 = N.Warstwy(x_train_set.shape[1], liczba_neuronow)
    layer1.forward(x_train_set)

    # używam pierwszej funkcji aktywacji
    aktywacja1 = N.Funkcje_aktywacji()
    aktywacja1.Aktywacja_ReLU(layer1.output)

    # tworze druga warstwe
    layer2 = N.Warstwy(liczba_neuronow, 2)
    layer2.forward(aktywacja1.output)

    # używam drugiej funkcji aktywacji
    aktywacja2 = N.Aktywacja_Softmax()
    aktywacja2.forward(layer2.output)

    ocena = N.Ocena()
    #print(layer1.wagi)


    lowest_lost = 1000
    best_layer1_weights = layer1.wagi.copy()
    best_layer1_biases = layer1.biasy.copy()
    best_layer2_weights = layer2.wagi.copy()
    best_layer2_biases = layer2.biasy.copy()

    for k in range(3):
        # uczenie sieci
        for i in range(1000):
            layer1.wagi =  np.random.randn(x_train_set.shape[1],liczba_neuronow)
            layer1.biasy =  np.random.randn(1, liczba_neuronow)
            layer2.wagi = np.random.randn(liczba_neuronow, 2)
            layer2.biasy = np.random.randn(1, 2)

            layer1.forward(x_train_set)
            aktywacja1.Aktywacja_ReLU(layer1.output)
            layer2.forward(aktywacja1.output)
            aktywacja2.forward(layer2.output)

            loss = ocena.calc_loss(aktywacja2.prawdopodobienstwa, y_train_set)
            accuracy = ocena.calc_accurcy(aktywacja2.prawdopodobienstwa, y_train_set)
            if loss < lowest_lost:
                print('epoka(iteracja):', i, ', loss:',loss, ', accuracy:', accuracy[0], ', sensitivity:', accuracy[2], ', specificity:' ,accuracy[1])

                best_layer1_weights = layer1.wagi.copy()
                best_layer1_biases = layer1.biasy.copy()
                best_layer2_weights = layer2.wagi.copy()
                best_layer2_biases = layer2.biasy.copy()
                lowest_lost = loss

    macierz = [loss, accuracy[0], accuracy[2], accuracy[1]]


    # sprawdzenie na zbiorze testowym
    layer1t = N.Warstwy(x_test_set.shape[1],liczba_neuronow)
    layer1t.wagi = best_layer1_weights
    layer1t.biasy = best_layer1_biases
    layer1t.forward(x_test_set)
    aktywacj1t = N.Funkcje_aktywacji()
    aktywacj1t.Aktywacja_ReLU(layer1t.output)
    layer2t = N.Warstwy(liczba_neuronow,2)
    layer2t.wagi = best_layer2_weights
    layer2t.biasy = best_layer2_biases
    layer2t.forward(aktywacj1t.output)
    aktywacja2t = N.Aktywacja_Softmax()
    aktywacja2t.forward(layer2t.output)

    ocena = N.Ocena()
    stratat = ocena.calc_loss(aktywacja2t.prawdopodobienstwa, y_test_set)
    ocenat = ocena.calc_accurcy(aktywacja2t.prawdopodobienstwa, y_test_set)
    print('\nZbiór testowy')
    print('loss:', stratat, ', accuracy:', ocenat[0], ', sensitivity:', ocenat[2],
          ', specificity:', ocenat[1])




# for i in range(10):
#     print(obserwacja[i])
#
#
# workbook = xlsxwriter.Workbook('bledy.xlsx')
# testowa = workbook.add_worksheet()
# uczaca = workbook.add_worksheet()
# row = 0
#
# for col, data in enumerate(obserwacja):
#     testowa.write_column(row, col, data)
#
# workbook.close()