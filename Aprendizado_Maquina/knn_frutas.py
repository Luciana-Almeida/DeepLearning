# Vamos começar estruturando os dados usando apenas listas puras do Python. Como você sugeriu antes, vamos focar em usar o valor de 𝑘 = 3.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. criadno tabela de dados (DataFrame)
dados = {
    'Peso': [120, 280, 150, 260, 130, 290],
    'Cor': [8, 2, 9, 3, 7, 1],
    'Classe': ['Maçã', 'Laranja', 'Maçã', 'Laranja', 'Maçã', 'Laranja']
}
df = pd.DataFrame(dados)

# 2. Separando as características (X) e a resposta (y)
X = df[['Peso', 'Cor']]
y = df[ 'Classe']

# 3. Normalizando os dados automaticamente
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)

# 4. Criando e treinando o modelo k-NN (com k = 3)
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_normalizado, y)

# 5. Preparando a nova fruta para previsão
nova_fruta = [[140, 8]]
nova_fruta_normalizada = scaler.transform(nova_fruta)

# 6. Fazendo previsão
previsao = modelo.predict(nova_fruta_normalizada)
print(f"A categoria prevista para a nova fruta é: {previsao[0]}")
