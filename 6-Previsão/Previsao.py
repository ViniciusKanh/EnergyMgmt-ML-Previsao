import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression
import plotly.express as px
from multiprocessing import Process

def sua_funcao():
    client = Client()

    df = dd.read_csv('0-Dataset/BigData_Consumo_Tratados.data')
    columns_for_prediction = ['Temperatura', 'Umidade', 'Velocidade do Vento']
    X = df[columns_for_prediction].to_dask_array(lengths=True)

    # Nome das colunas alvo
    target_columns = ['Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3']
    X_train = X
    X_test = X 

    future_predictions = {}

    for target_col in target_columns:
        y = df[target_col].to_dask_array(lengths=True)
        y_train = y
        model = LinearRegression()
        model.fit(X_train, y_train)
        future_pred = model.predict(X_test).compute()
        future_predictions[target_col] = future_pred

    test_datetime = pd.date_range(start='2018-01-01', periods=X_test.shape[0], freq='H')

    data_melted_list = []

    for target_col in target_columns:
        data_melted = pd.DataFrame({
            'DateTime': test_datetime,
            'Power Consumption': future_predictions[target_col],
            'Zone': target_col
        })
        data_melted_list.append(data_melted)

    data_melted_all = pd.concat(data_melted_list)

    fig = px.line(data_melted_all,
                x='DateTime',
                y='Power Consumption',
                color='Zone',
                title='Previsão de Consumo de Energia para 2018',
                labels={'Power Consumption': 'Consumo de Energia', 'DateTime': 'Data e Hora'},
                template='plotly_dark')
    fig.show()
    fig.write_html("6-Previsão/PrevisaoBigDataConsumo2018.html")

if __name__ == '__main__':
    p = Process(target=sua_funcao)
    p.start()
    p.join()
