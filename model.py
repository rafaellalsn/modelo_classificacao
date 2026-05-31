'''
Import de Bibliotecas
'''
import pandas as pd #Biblioteca para manipulação de dados
import numpy as np #Biblioteca importante para cálculos
import matplotlib.pyplot as plt #Bibliotecas para visualização de dados
import seaborn as sns

from sklearn.model_selection import train_test_split #Partição dos dados em treino e teste
from sklearn.ensemble import RandomForestClassifier #Modelo que combina Árvores de Decisão
from sklearn.tree import DecisionTreeClassifier, plot_tree #Modelo baseado é única árvore
from sklearn.metrics import classification_report, accuracy_score #Importando a taxa de acerto e o resumo de classificação
import pickle

'''
Leitura dos dados
'''
df = pd.read_csv(r'instances/medicamentos.csv', sep=';', encoding='latin1')

'''
Exibir alguns resumos dos dados
'''
print(df.head(8)) #Comando head() é usado para retornar as primeiras observações dos dados

print(df.shape) #Retorna o número de linhas e o número de colunas

print(df['Medicamento'].value_counts()) #Retorna a frequencia de valores de uma variável

print(df.isnull().sum()) #Retorna a quantidade de dados faltantes

print(df.duplicated().sum()) #Retorna a quantidade de dados duplicados


'''
Visualização de dados
'''
sns.countplot(data = df, x = 'Medicamento', hue = 'Sexo')
plt.show()

sns.countplot(data = df, x = 'Medicamento', hue = 'Pressão')
plt.show()

'''
Pré-processamento: Transformar as classes das variáveis em representação numérica
'''
df['Pressão'] = df['Pressão'].map({'Baixo' : 0, 'Normal' : 1, 'Alto': 2})
df['Colesterol'] = df['Colesterol'].map({'Normal' : 1, 'Alto' : 2})
df['Sexo'] = df['Sexo'].map({'M' : 0, 'F' : 1})
df['Medicamento'] = df['Medicamento'].map({'Y' : 0, 'X' : 1, 'A' : 2, 'B' : 3, 'C' : 4})
print(df.head())

'''
Início da configuração de treino
'''
X = df.drop('Medicamento', axis = 1)   # X é o conjunto de variaveis explicativas (caracteristicas)
Y = df['Medicamento']  # Rótulo ou a variável resposta

#Função para particionar os dados em treino e teste 
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=0)

#Técnica 1
DT = DecisionTreeClassifier() #Técnica de Árvore de Decisão
DT.fit(X_train, y_train) #O comando fit recebe x e y (treino) para aprender os padrões
DT_pred = DT.predict(X_test) #O comando predict retorna predições, ou seja, respostas

#Técnica 2
RF = RandomForestClassifier(n_estimators=100) #Técnica de Combinação de Árvres
RF.fit(X_train, y_train) #O comando fit recebe x e y (treino) para aprender os padrões
RF_pred = RF.predict(X_test) #O comando predict retorna predições, ou seja, respostas

'''
Os dados y_test são usados como um gabarito para verificar o quanto os modelos acertaram
comparando o valor verdadeiro (y_test) com o valor predito pelos modelos
'''
print("Árvore de Decisão:",accuracy_score(y_test, DT_pred)) # Calcula a taxa de acerto
print("Random Forest:",accuracy_score(y_test, RF_pred))

'''
Exibe a estrutura do modelo de Árvore de Decisão
'''
plt.figure(figsize=(12, 8))
caracteristicas_names = X.columns.tolist()
rotulo_name = Y.unique().astype(str).tolist()
plot_tree(DT, 
          feature_names=caracteristicas_names,  # Nomes das características
          class_names=rotulo_name,    # Nomes das classes do rótulo
          filled=True,                      # Preencher com cores
          rounded=True)                     # Nós arredondados
plt.show()


# Salvando o modelo treinado e criando um arquivo
with open('modelo_preditivo.pkl', 'wb') as f:
    pickle.dump(DT, f)