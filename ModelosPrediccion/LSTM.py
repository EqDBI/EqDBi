import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

def ejecutar_lstm(df):
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

    # Definir y entrenar el modelo LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model_lstm.add(LSTM(units=50))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    model_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Hacer predicciones
    predictions_lstm = model_lstm.predict(X_test)
    predictions_lstm = scaler_target.inverse_transform(predictions_lstm.reshape(-1, 1)).flatten()
    y_test_lstm = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Asegurarse de que test_dates, predictions_lstm y y_test_lstm tengan la misma longitud
    test_dates = df.index[-len(predictions_lstm):]

    # Crear un DataFrame para las predicciones y los valores reales en el conjunto de prueba
    predictions_df_lstm = pd.DataFrame({
        'Fecha': test_dates,
        'Predicciones LSTM': predictions_lstm,
        'Valores Reales': y_test_lstm
    })

    # Graficar las predicciones y los valores reales del conjunto de prueba para LSTM
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_lstm['Fecha'],
        y=predictions_df_lstm['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_lstm['Fecha'],
        y=predictions_df_lstm['Predicciones LSTM'],
        mode='lines',
        name='Predicciones LSTM',
        line=dict(color='blue')
    ))
    fig_pred.update_layout(
        title='Predicciones LSTM vs Valores Reales del Precio de Cierre (Conjunto de Prueba)',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    # Evaluación del modelo
    mse = mean_squared_error(y_test_lstm, predictions_lstm)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_lstm, predictions_lstm)

    return mse, rmse, mape, fig_pred, predictions_lstm[-1]
