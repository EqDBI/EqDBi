# Basicos
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Evaluacion
from sklearn.metrics import mean_squared_error

# Visualizaciones
import matplotlib.pyplot as plt
import seaborn as sns

# Support Vector Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline

from .preprocessing import preprocessing

def ejecutar_svr(df):

    # Seleccionar las columnas que quieres usar como input para el modelo SVR
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Precio Anterior',
                'Precio Máximo Anterior', 'Precio Mínimo Anterior', 'Precio Apertura Anterior',
                'PM_10', 'Middle Band Bollinger', 'Upper Band Bollinger',
                'Lower Band Bollinger', 'Precio Medio', 'Precio Plata']

    X = df[features]
    y = df['Precio Siguiente']

    # División de los datos en conjuntos de entrenamiento y prueba
    split = int(0.8 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Definición de los parámetros para GridSearchCV
    param_grid = {
        'svr__kernel': ['poly'],
        'svr__degree': [1, 2],
        'svr__C': [2**-1, 1, 2, 3, 10, 20, 100]
    }

    # Crear un pipeline que incluya la estandarización y el modelo SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Configurar GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', error_score='raise')

    # Entrenar el modelo
    grid_search.fit(X_train, y_train)

    # Obtener los mejores parámetros y el mejor modelo
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Mostrar los mejores parámetros
    print("Mejores parámetros:", best_params)

    # Realizar predicciones con el conjunto de prueba
    y_pred = best_model.predict(X_test)

    # Crear un DataFrame con las fechas y los valores de prueba y predicciones
    result_df = pd.DataFrame({'Fecha': X_test.index, 'Valores Reales': y_test, 'Predicciones': y_pred})

    # Graficar las predicciones vs los valores reales
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=result_df['Fecha'], y=result_df['Valores Reales'], mode='lines', name='Valores Reales'))
    fig_pred.add_trace(go.Scatter(x=result_df['Fecha'], y=result_df['Predicciones'], mode='lines', name='Predicciones', line=dict(color='red', dash='dash')))
    fig_pred.update_layout(
        title='Comparación de Predicciones y Valores Reales',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )
    fig_pred.update_xaxes(tickangle=45)

    # Mostrar las figuras de predicción
    st.plotly_chart(fig_pred)
    
    # Evaluación del modelo
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean((np.abs(y_test - y_pred) / y_test)) * 100

    # Mostrar métricas de evaluación
    st.subheader("Métricas de Evaluación")
    st.write(f"Error Cuadrático Medio (MSE): {mse}")
    st.write(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
    st.write(f"Error Absoluto Porcentual Promedio (MAPE): {mape}")


    return mse, rmse, mape, fig_pred