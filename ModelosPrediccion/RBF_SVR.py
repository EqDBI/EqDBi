import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go

def ejecutar_rbf_svr(df):
    # Seleccionar las columnas que quieres usar como input para el modelo
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Precio Anterior',
                'Precio Máximo Anterior', 'Precio Mínimo Anterior', 'Precio Apertura Anterior',
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

    # Definir y entrenar el modelo híbrido RBF + SVR
    rbf_feature = RBFSampler(gamma=0.01, n_components=500, random_state=1)
    svm = SVR(kernel='linear')
    model_rbf = make_pipeline(rbf_feature, svm)
    model_rbf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Hacer predicciones
    predictions_rbf = model_rbf.predict(X_test.reshape(X_test.shape[0], -1))

    # Invertir la escala de las predicciones y de y_test
    predictions_rbf = scaler_target.inverse_transform(predictions_rbf.reshape(-1, 1)).flatten()
    y_test_rbf = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Asegurarse de que test_dates, predictions_rbf y y_test_rbf tengan la misma longitud
    test_dates = df.index[-len(predictions_rbf):]

    # Crear un DataFrame para las predicciones y los valores reales en el conjunto de prueba
    predictions_df_rbf = pd.DataFrame({
        'Fecha': test_dates,
        'Predicciones RBF': predictions_rbf,
        'Valores Reales': y_test_rbf
    })

    # Graficar las predicciones y los valores reales del conjunto de prueba para RBF
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_rbf['Fecha'],
        y=predictions_df_rbf['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))
    fig_pred.add_trace(go.Scatter(
        x=predictions_df_rbf['Fecha'],
        y=predictions_df_rbf['Predicciones RBF'],
        mode='lines',
        name='Predicciones RBF',
        line=dict(color='red')
    ))
    fig_pred.update_layout(
        title='Predicciones RBF vs Valores Reales del Precio de Cierre (Conjunto de Prueba)',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    # Evaluación del modelo
    mse = mean_squared_error(y_test_rbf, predictions_rbf)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_rbf, predictions_rbf)

    return mse, rmse, mape, fig_pred, predictions_rbf[-1]
