import pandas as pd
import seaborn as pl
import matplotlib.pyplot as plt


def Wczytywanie_danych():
    #wczytywanie danych
    data = pd.read_excel(r'titanic.xlsx')

    #wstępna analiza
    #data.info()

    #brakujące wartości
    data.isnull().sum().to_frame()
    data = data.dropna(subset=['Embarked'])
    #px.box(data.Age).show()
    srednia_wieku = data.Age.mean()
    data.Age = data.Age.fillna(srednia_wieku)

    #usuniecie niepotrzebnych zmiennych
    data = data.drop("Name", axis = 1)
    data = data.drop("PassengerId", axis = 1)
    data = data.drop("Ticket", axis = 1)
    data = data.drop("Cabin", axis = 1)

    #zmienne kategoryczne
    data_sex = pd.get_dummies(data['Sex'])
    data_embarked = pd.get_dummies(data['Embarked'])
    data = data.drop('Sex', axis = 1)
    data = data.drop('Embarked', axis = 1)
    data_final = pd.concat([data, data_sex, data_embarked], axis = 1)

    #normalizacja danych
    data_final['Age'] -= min(data_final['Age'])
    data_final['Age'] = data_final['Age']/(max(data_final['Age'])-min(data_final['Age']))
    data_final['Fare'] -= min(data_final['Fare'])
    data_final['Fare'] = data_final['Fare']/(max(data_final['Fare'])-min(data_final['Fare']))
    return data_final

# data_before = pd.read_excel(r'titanic.xlsx')
# data = Wczytywanie_danych()
#chart1 = pl.countplot(x = 'Survived', data = data, )
#chart2 = pl.countplot(x = 'Survived',hue = "Parch", data = data_before)
#pl.set_style('whitegrid')
#plt.show()
#data1 = data[1:178]
#chart1 = pl.countplot(x = 'Survived', data = data1 )
#plt.show()