#Importando Bibliotecas

import pandas as pd
import plotly.express as px

"""Carregando o Dataset

Inicialmente, os dados são carregados em um DataFrame pandas a partir de um arquivo CSV. O método read_csv é utilizado para este propósito.
"""

df = pd.read_csv('0-Dataset/BigData_Consumo_Tratados.data')
print(df.columns)

"""Convertendo a coluna 'DateTime' para o tipo datetime

O próximo passo é a conversão da coluna 'DateTime' para o tipo de dado apropriado, isto é, datetime. Isso facilita operações e visualizações subsequentes que dependem da ordenação temporal correta.
"""

df['DateTime'] = pd.to_datetime(df['DateTime'])

"""Reformatando o DataFrame para o formato longo usando melt

Usando o método melt do pandas, o DataFrame é transformado do formato largo para o longo, permitindo que cada linha represente uma observação de consumo de energia para uma zona específica em um ponto específico no tempo. Este formato é mais adequado para a visualização desejada.
"""

data_melted = df.melt(id_vars='DateTime',
                      value_vars=['Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3'],
                      var_name='Zone', value_name='Power Consumption')

"""Criando a animação

O gráfico de linhas é criado usando plotly.express. A dimensão x representa o tempo, enquanto a y representa o consumo de energia. A coloração diferenciada para cada zona permite a distinção das três séries temporais. A interatividade oferecida por plotly permite que, ao passar o mouse sobre o gráfico, o usuário obtenha informações detalhadas sobre o consumo de energia para uma zona específica em um determinado ponto no tempo.

Por fim, o gráfico é salvo como uma página HTML para posterior visualização ou compartilhamento.
"""

fig = px.line(data_melted,
              x='DateTime',
              y='Power Consumption',
              color='Zone',
              title='Consumo de Energia ao Longo do Tempo',
              labels={'Power Consumption': 'Consumo de Energia', 'DateTime': 'Data e Hora'},
              template='plotly_dark')

fig.show()

fig.write_html("3-DataVisualization/DadosAtemporal.html")