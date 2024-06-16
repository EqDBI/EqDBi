#C:\Users\PROGRESA HUACHIPA\OneDrive\Escritorio\BISem12\BuenaAventura\MLP.py
# -*- coding: utf-8 -*-
"""Eq.D_Buenaventura

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TwldTMHFwQpVQarDqC2pmdoZsCe54Vvr

# IMPORTING LIBRARIES
"""

# Basicos
import pandas as pd # Manipulación y análisis de datos
import numpy as np # Operaciones numéricas y algebra lineal

# Extraer información del instrumento financiero
import yfinance as yf # Descarga de datos financieros desde Yahoo Finance

# Preprocesamiento
from sklearn.model_selection import train_test_split # División de los datos en conjuntos de entrenamiento y prueba
from sklearn.preprocessing import StandardScaler # Escalado de características
from sklearn.preprocessing import MinMaxScaler # Escalado de características a un rango específico
from statsmodels.tsa.seasonal import seasonal_decompose # Descomposición de series temporales

# Evaluacion
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Métricas de evaluación

# Visualizaciones
import matplotlib.pyplot as plt # Generación de gráficos y visualizaciones
import seaborn as sns # Visualización de datos estadísticos

# RNN: LSTM
from tensorflow.keras.models import Sequential # Creación de modelos secuenciales de Keras
from tensorflow.keras.layers import LSTM, Dense # Capas LSTM y densas para redes neuronales

# Support Vector Regressor
from sklearn.model_selection import GridSearchCV # Búsqueda en cuadrícula para optimización de hiperparámetros
from sklearn.svm import SVR # Soporte vectorial para regresión
from sklearn.pipeline import Pipeline # Creación de pipelines para flujos de trabajo de ML

# ANN: MLP Regressor
from sklearn.neural_network import MLPRegressor # Regressor de perceptrón multicapa

# Modelo Híbrido
from sklearn.kernel_approximation import RBFSampler # Aproximación de kernel de base radial (RBF)
from sklearn.linear_model import Ridge # Regresión Ridge
from sklearn.pipeline import make_pipeline # Creación de pipelines para flujos de trabajo de ML

from ModelosPrediccion.extraccion_datos import extraccion_datos

import plotly.graph_objects as go


"""# COMPAÑIA DE MINAS BUENAVENURA SAA (BVN)

## Extracción de datos
"""
def ejecutar_mlp(df,instrumento_financiero, fecha_inicio, fecha_fin):
    bvn_df, fig1, fig2, fig3, fig4 = extraccion_datos(df,instrumento_financiero, fecha_inicio, fecha_fin)



    """## MODELO: Red Neuronal Artificial (MLP Regressor)

    """

    # Seleccionar las columnas que quieres usar como input para el modelo
    features = bvn_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Precio Anterior',
                    'Precio Máximo Anterior', 'Precio Mínimo Anterior', 'Precio Apertura Anterior',
                    'PM_10', 'Middle Band Bollinger', 'Upper Band Bollinger',
                    'Lower Band Bollinger', 'Precio Medio', 'Precio Plata']].values

    # Escalar los datos
    # Se instancia el escalador MinMaxScaler para escalar las características entre 0 y 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Se ajusta el escalador a los datos y se transforman las características
    scaled_features = scaler.fit_transform(features)

    # Definir la longitud de la secuencia (número de timesteps)
    # Se define la cantidad de pasos temporales que se considerarán para cada muestra
    timesteps = 60

    # Preparar los datos para el modelo
    # Inicialización de las listas para almacenar las secuencias de entrada y los valores de salida
    X = []
    y = []

    # Crear secuencias de datos para entrenar el modelo
    # Se itera sobre el rango de datos, creando secuencias de 'timesteps' longitud
    for i in range(timesteps, len(scaled_features)):
        # Se añaden los 'timesteps' datos anteriores a la lista de entradas X
        X.append(scaled_features[i-timesteps:i])
        # Se añade el precio de cierre actual (índice 3) a la lista de salidas y
        y.append(scaled_features[i, 3])

    # Convertir las listas a arrays numpy para que puedan ser utilizados por los modelos de ML
    X, y = np.array(X), np.array(y)

    # Divide en conjuntos de entrenamiento y prueba
    # Se define el índice de división para el 80% de los datos como conjunto de entrenamiento
    split = int(0.8 * len(X))
    # Se dividen los datos en conjuntos de entrenamiento y prueba
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    """### Definicion del modelo"""

    # Definir y entrenar el modelo MLP
    model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)
    model_mlp.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Hacer predicciones
    predictions_mlp = model_mlp.predict(X_test.reshape(X_test.shape[0], -1))

    # Invertir la escala de las predicciones y de y_test
    predictions_mlp = scaler.inverse_transform(np.concatenate((predictions_mlp.reshape(-1, 1), np.zeros((predictions_mlp.shape[0], scaled_features.shape[1]-1))), axis=1))[:,0]
    y_test_mlp = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_features.shape[1]-1))), axis=1))[:,0]

    # Asegurarse de que test_dates, predictions_mlp y y_test_mlp tengan la misma longitud
    test_dates = bvn_df.index[-len(predictions_mlp):]

    # Crear un DataFrame para las predicciones y los valores reales en el conjunto de prueba
    predictions_df_mlp = pd.DataFrame({
        'Fecha': test_dates,
        'Predicciones MLP': predictions_mlp,
        'Valores Reales': y_test_mlp
    })

    # Graficar las predicciones y los valores reales del conjunto de prueba para MLP
    fig5 = go.Figure()

# Añadir la línea de Valores Reales
    fig5.add_trace(go.Scatter(
        x=predictions_df_mlp['Fecha'],
        y=predictions_df_mlp['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))

    # Añadir la línea de Predicciones MLP
    fig5.add_trace(go.Scatter(
        x=predictions_df_mlp['Fecha'],
        y=predictions_df_mlp['Predicciones MLP'],
        mode='lines',
        name='Predicciones MLP',
        line=dict(color='blue')
    ))

    # Configurar el diseño de la figura
    fig5.update_layout(
        title='Predicciones MLP vs Valores Reales del Precio de Cierre (Conjunto de Prueba)',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    """### Evaluación del modelo"""

    # Calcular las métricas de evaluación
    mse = mean_squared_error(y_test_mlp, predictions_mlp)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_mlp, predictions_mlp)

    return mse, rmse, mape, fig1, fig2, fig3, fig4, fig5
