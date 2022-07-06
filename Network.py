import numpy as np

# TWORZENIE SIECI NEURONOWEJ
class Warstwy:
    def __init__(self, rozmiar_inputu, liczba_neuronow):
        #losuje macierz z wagami, tak żeby wartości były od -1 do 1
        self.wagi = 0.05 * np.random.randn(rozmiar_inputu, liczba_neuronow)
        #ustawiam macierz z biasami na zera
        self.biasy = np.zeros((1, liczba_neuronow))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.wagi) + self.biasy

class Funkcje_aktywacji:
    def Aktywacja_ReLU(self, inputs):
        self.output = np.maximum(0, inputs)
    def Aktywacja_tanh(self, inputs):
        self.output = np.tanh(inputs)
    def Aktywacja_liniowa(self, inputs):
        self.output = inputs
    def Aktywacja_sigmoid(self, inputs):
        self.output = 1/(1 + np.exp(-inputs))

class Aktywacja_Softmax:
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
        self.prawdopodobienstwa = exp / np.sum(exp, axis = 1, keepdims = True)

# [class1=1,class2=0]
class Ocena:
    def calc_loss(self, output, real_values):
        y_pred_clipped = np.clip(output, 1e-7, 1-1e-7)
        correct_confidences = np.sum(y_pred_clipped * real_values, axis = 1)
        negative_log = -np.log(correct_confidences)
        data_loss = np.mean(negative_log)
        return(data_loss)

    def calc_accurcy(self, y_pred, y_real):
        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0
        for i, j in zip(y_pred, y_real):
            if i[0] < 0.5:
                i[0] = 0
            elif i[0] > 0.5:
                i[0] = 1

            if i[1] < 0.5:
                i[1] = 0
            elif i[1] > 0.5:
                i[1] = 1

            if i[0] == 1 == j[0]:
                true_positive += 1
            elif i[0] == 0 == j[0]:
                true_negative += 1
            elif i[0] == 0 != j[0]:
                false_positive += 1
            elif i[0] == 1 != j[0]:
                false_negative += 1

        accuracy = (true_positive + true_negative) / len(y_pred)
        Sensitivity = true_positive / (true_positive + false_positive)
        specificity = true_negative / (true_negative + false_negative)
        matrix = [accuracy,specificity, Sensitivity]
        return matrix