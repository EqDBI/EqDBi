# Basicos
import pandas as pd
import numpy as np

# Preprocesamiento
from sklearn.preprocessing import MinMaxScaler

# Evaluacion
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ANN: MLP Regressor
from sklearn.neural_network import MLPRegressor
import plotly.graph_objects as go


def ejecutar_mlp(df):

    # Seleccionar las columnas que quieres usar como input para el modelo
    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'PM_10', 'Middle Band Bollinger', 'Upper Band Bollinger',
                'Lower Band Bollinger', 'Precio Medio', 'Precio Plata']

    # Escalar los datos
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df[features])

    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(df[['Precio Siguiente']])

    # Definir la longitud de la secuencia (número de timesteps)
    timesteps = 60

    # Preparar los datos para el modelo
    X_seq = []
    y_seq = []

    # Crear las secuencias de datos para las características y el objetivo
    for i in range(timesteps, len(scaled_features)):
        # Crear una ventana deslizante de longitud `timesteps` para las características
        X_seq.append(scaled_features[i-timesteps:i])
        # Añadir el valor del precio del día siguiente de la columna escalada 'Precio Siguiente' a y_seq
        y_seq.append(scaled_target[i, 0])

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    # Divide en conjuntos de entrenamiento y prueba
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Definir y entrenar el modelo MLP
    model_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=1)
    model_mlp.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Hacer predicciones
    predictions_mlp = model_mlp.predict(X_test.reshape(X_test.shape[0], -1))

    # Invertir la escala de las predicciones y de y_test
    predictions_mlp = scaler_target.inverse_transform(predictions_mlp.reshape(-1, 1)).flatten()
    y_test_mlp = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Asegurarse de que test_dates, predictions_mlp y y_test_mlp tengan la misma longitud
    test_dates = df.index[-len(predictions_mlp):]

    # Crear un DataFrame para las predicciones y los valores reales en el conjunto de prueba
    predictions_df_mlp = pd.DataFrame({
        'Fecha': test_dates,
        'Predicciones MLP': predictions_mlp,
        'Valores Reales': y_test_mlp
    })

    # Graficar las predicciones y los valores reales del conjunto de prueba para MLP
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_mlp['Fecha'],
        y=predictions_df_mlp['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_mlp['Fecha'],
        y=predictions_df_mlp['Predicciones MLP'],
        mode='lines',
        name='Predicciones MLP',
        line=dict(color='blue')
    ))
    fig_pred.update_layout(
        title='Predicciones MLP vs Valores Reales del Precio de Cierre (Conjunto de Prueba)',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    # Evaluación del modelo
    mse = mean_squared_error(y_test_mlp, predictions_mlp)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_mlp, predictions_mlp)


    return mse, rmse, mape, fig_pred, predictions_mlp[-1]
