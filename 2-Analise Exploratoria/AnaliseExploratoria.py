# Importando Bilbiotecas necessarias

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""Carregando a base de dados"""

df = pd.read_csv('0-Dataset/BigData_Consumo_Tratados.data')

"""Visualizar primeiras linhas de dados"""

print(df.head())

"""## Histogramas para entender a distribuição dos dados"""

features = list(df.columns)

plt.figure(figsize=(15, 15))
for idx, feature in enumerate(features[1:], 1):
    plt.subplot(len(features[1:]), 1, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histograma para {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequência')
    plt.grid(True)

plt.tight_layout()
plt.show()

"""### Mapa de Calor - Correlaçao"""

correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Calor da Matriz de Correlação')
plt.show()

# Amostragem dos dados para agilizar o processo (opcional, mas recomendado para grandes datasets)
df_sample = df.sample(frac=0.1)

# Todas as colunas numéricas para a comparação
all_columns = ['Temperatura', 'Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3']

# Cálculo do número de plots
n = len(all_columns)
total_plots = n * (n - 1) // 2

# Configuração da figura
plt.figure(figsize=(24, 24))  # Ajuste o tamanho da figura aqui

plot_count = 1  # Contador para o número do subplot

# Loop aninhado para gerar gráficos de dispersão para cada par de colunas
for i in range(len(all_columns)):
    for j in range(i + 1, len(all_columns)):
        plt.subplot(n, n, plot_count)
        sns.scatterplot(data=df_sample, x=all_columns[i], y=all_columns[j])
        plt.title(f'Gráfico de Dispersão entre {all_columns[i]} e {all_columns[j]}')
        plt.grid(True)
        plot_count += 1

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Ajuste o espaço entre os subplots aqui
plt.show()



