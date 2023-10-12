"""
Importação das bibliotecas
"""
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load

def main():
    """
    Inicialização do Cliente Dask
    """
    client = Client()
    print(client)

    """
    Carregamento dos Dados
    """
    df = dd.read_csv('0-Dataset/BigData_Consumo_Tratados.data', delimiter=',')

    """
    Exploração Inicial dos Dados
    """
    print(df.head())

    """
    Preparação das variáveis independentes (X)
    """
    X = df[['Temperatura', 'Umidade', 'Velocidade do Vento', 'fluxos difusos gerais', 'fluxos difusos']].to_dask_array(lengths=True)

    """
    Modelagem e Avaliação
    """
    models = {}
    metrics = {}
    target_columns = ['Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3']

    for target_col in target_columns:
        y = df[target_col].to_dask_array(lengths=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Salve o modelo treinado
        dump(model, f'0-Modelos/{target_col}_model.joblib')
        
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        metrics[target_col] = mse
        print(f"Erro Quadrático Médio para {target_col}: {mse}")

    print("Métricas para todas as zonas:", metrics)

    """
    Visualização
    """
    for target_col in target_columns:
        y_subset = df[target_col].compute().values[:1000]
        X_subset = X.compute()[:1000, :]

        X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.2, shuffle=True, random_state=0)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        corr_coef = np.corrcoef(y_test, y_pred)[0, 1]

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, label='Pontos de Dados')
        plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', label='Linha de Melhor Ajuste')
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title(f'Valores Reais vs Previsões para {target_col} \n Coeficiente de Correlação: {corr_coef:.2f}')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()
