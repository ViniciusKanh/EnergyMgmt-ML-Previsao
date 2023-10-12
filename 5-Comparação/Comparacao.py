## Processo 5 - Comparação de Modelos

# Importação de Bibliotecas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Carregamento dos Dados
df = pd.read_csv('0-Dataset/BigData_Consumo_Tratados.data', delimiter=',')

# Remove a coluna de data e hora e outras colunas dependentes
X_expanded = df.drop(['DateTime', 'Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3'], axis=1)

# Listas para métricas
models = ['Regressão Linear', 'Árvore de Decisão', 'Random Forest']
mse_values = []
mae_values = []
r2_values = []

# Treinamento e Avaliação de Modelos para cada zona
target_columns = ['Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3']

for target_col in target_columns:
    y = df[target_col]
    X_train_exp, X_test_exp, y_train, y_test = train_test_split(X_expanded, y, test_size=0.2, random_state=0)

    # Regressão Linear
    model_linear = LinearRegression()
    model_linear.fit(X_train_exp, y_train)
    y_pred_linear = model_linear.predict(X_test_exp)
    mse_values.append(mean_squared_error(y_test, y_pred_linear))
    mae_values.append(mean_absolute_error(y_test, y_pred_linear))
    r2_values.append(r2_score(y_test, y_pred_linear))

    # Árvore de Decisão
    model_tree = DecisionTreeRegressor()
    model_tree.fit(X_train_exp, y_train)
    y_pred_tree = model_tree.predict(X_test_exp)
    mse_values.append(mean_squared_error(y_test, y_pred_tree))
    mae_values.append(mean_absolute_error(y_test, y_pred_tree))
    r2_values.append(r2_score(y_test, y_pred_tree))

    # Random Forest
    model_rf = RandomForestRegressor()
    model_rf.fit(X_train_exp, y_train)
    y_pred_rf = model_rf.predict(X_test_exp)
    mse_values.append(mean_squared_error(y_test, y_pred_rf))
    mae_values.append(mean_absolute_error(y_test, y_pred_rf))
    r2_values.append(r2_score(y_test, y_pred_rf))

# Plotar Gráfico Comparativo
x = np.arange(len(models) * len(target_columns)) # Multiplica pela quantidade de zonas
width = 0.25

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plotar MSE
ax1.bar(x - width, mse_values, width, color='b', label='MSE')

# Plotar MAE
ax1.bar(x, mae_values, width, color='g', label='MAE')

# Configurações de eixo y para MSE e MAE
ax1.set_ylabel('MSE / MAE')
ax1.set_xlabel('Modelos por Zona')
ax1.set_xticks(x)
ax1.set_xticklabels(models * len(target_columns), rotation=45) # Multiplica pelos nomes das zonas

# Adicionar legendas
ax1.legend(loc='upper left', bbox_to_anchor=(1,1))

ax2 = ax1.twinx()

# Plotar R2
ax2.bar(x + width, r2_values, width, color='r', label='R2')

# Configurações de eixo y para R2
ax2.set_ylabel('R2')

# Adicionar legendas
ax2.legend(loc='upper left', bbox_to_anchor=(1,0.9))

# Título
plt.title('Comparação de Desempenho dos Modelos por Zona')

fig.tight_layout()
plt.show()
