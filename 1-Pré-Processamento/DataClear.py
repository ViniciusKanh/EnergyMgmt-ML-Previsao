"""## Processo 1 - Pré Processamento

Importando Bibiotencas
"""

import pandas as pd
import numpy as np
from missingno import matrix

"""Definindo Funçoes para o Pré-Processamento"""

def transform_consumo_to_float(df, column_name):
    df[column_name] = df[column_name].str.replace(',', '.').astype(float)
    return df

def UpdateMissingValues(df, column, method="median", number=0):
    if method == 'number':
        df[column].fillna(number, inplace=True)
    elif method == 'median':
        median = round(df[column].median(), 2)
        df[column].fillna(median, inplace=True)
    elif method == 'mean':
        mean = round(df[column].mean(), 2)
        df[column].fillna(mean, inplace=True)
    elif method == 'mode':
        mode = df[column].mode()[0]
        df[column].fillna(mode, inplace=True)

def verify_invalid_values(df):
    # Exemplo: Verificar se há algum valor negativo na coluna 'Temperatura'
    if (df['Temperatura'] < 0).any():
        print("Valores inválidos encontrados na coluna 'Temperatura'")
        # Aqui você pode adicionar um tratamento específico, por exemplo, substituir por NaN
        df['Temperatura'] = df['Temperatura'].apply(lambda x: np.nan if x < 0 else x)
    return df

"""Detecção e Exclusão de Outliers na coluna 'Consumo de energia Zona 1'"""

def get_outliers_indices(df, col_name):
    if df[col_name].dtype not in ['int64', 'float64']:  # Verifica o tipo de dados
        return []
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_indices = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)].index
    return outliers_indices

"""Defindo Arquivos dos DATASETS"""

input_file =  '0-Dataset/BigData_Consumo_Brutos.data'
output_file = '0-Dataset/BigData_Consumo_Tratados.data'

# Leitura do conjunto de dados
df = pd.read_csv(input_file)

"""Filtragem de dados"""

zona_columns = ['Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3']
for zona in zona_columns:
    df = df[df[zona] > 0]

"""Detecção e exclusão de outliers"""

cols_for_outliers = ['Temperatura', 'Umidade'] + zona_columns
all_outliers_indices = []
for col in cols_for_outliers:
    outlier_indices = get_outliers_indices(df, col)
    if outlier_indices.size > 0:
        print(f"Outliers encontrados na coluna {col}.")
        all_outliers_indices.extend(outlier_indices.tolist())

# Remoção de outliers
all_outliers_indices = list(set(all_outliers_indices))
df.drop(index=all_outliers_indices, inplace=True)

"""Informações gerais dos dados"""

print("INFORMAÇÕES GERAIS DOS DADOS\n")
print(df.info())
print("\n")

"""Verificar Outliers em mais colunas e excluir."""

from scipy import stats

# Colunas para verificar outliers
cols_for_outliers = ['Temperatura', 'Umidade', 'Velocidade do Vento', 'fluxos difusos gerais', 'fluxos difusos','Consumo de energia da Zona 1', 'Consumo de energia da Zona 2', 'Consumo de energia da Zona 3'] + zona_columns

# Encontra os índices dos outliers e os remove
for col in cols_for_outliers:
    z_scores = stats.zscore(df[col])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)  # Aqui, 3 é o valor limite do Z-score
    df = df[filtered_entries]

"""Descrição dos dados"""

print("DESCRIÇÃO DOS DADOS\n")
print(df.describe())
print("\n")

"""Tratamento para registros faltantes"""

columns_missing_value = df.columns[df.isnull().any()]
method = 'mean'
for c in columns_missing_value:
    UpdateMissingValues(df, c, method)

"""Detecção de outliers nas zonas"""

for zona in zona_columns:
    outlier_indices = get_outliers_indices(df, zona)
    if outlier_indices.size > 0:
        print(f"Outliers encontrados na coluna {zona}: {outlier_indices.size} registros.")
        df.drop(index=outlier_indices, inplace=True)

"""Verificar e tratar valores inválidos ou inconsistentes"""

df = verify_invalid_values(df)

"""Salvar arquivo tratado"""

df.to_csv(output_file, header=True, index=False)