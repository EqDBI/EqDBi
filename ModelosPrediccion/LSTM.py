

# IMPORTING LIBRARIES


# Basicos
import pandas as pd # Manipulación y análisis de datos
import numpy as np # Operaciones numéricas y algebra lineal


from sklearn.preprocessing import MinMaxScaler # Escalado de características a un rango específico

# Evaluacion
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Métricas de evaluación

# Visualizaciones

import plotly.graph_objects as go

# RNN: LSTM
from tensorflow.keras.models import Sequential # type: ignore # Creación de modelos secuenciales de Keras
from tensorflow.keras.layers import LSTM, Dense # type: ignore # Capas LSTM y densas para redes neuronales


from ModelosPrediccion.preprocessing import preprocessing




"""# COMPAÑIA DE MINAS BUENAVENURA SAA (BVN)

## Extracción de datos
"""

def ejecutar_lstm(df,instrumento_financiero, fecha_inicio, fecha_fin):
        # Descargar datos históricos de Buenaventura en un rango de fechas específico
    bvn_df, fig1, fig2, fig3, fig4 = preprocessing(df,instrumento_financiero, fecha_inicio, fecha_fin)
    
    """## MODELO: Red Neuronal Recurrente Long Short Term Memory (LSTM)

    Modelado de X e y
    """


    # Seleccionar las columnas que quieres usar como input para el modelo LSTM
    features = bvn_df[[ 'Precio Anterior',
                    'Precio Máximo Anterior', 'Precio Mínimo Anterior', 'Precio Apertura Anterior',
                    'PM_10', 'Middle Band Bollinger', 'Upper Band Bollinger',
                    'Lower Band Bollinger', 'Precio Plata']].values

    # Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Definir la longitud de la secuencia (número de timesteps)
    timesteps = 60

    # Prepara los datos para la LSTM
    X = []
    y = []

    for i in range(timesteps, len(scaled_features)):
        X.append(scaled_features[i-timesteps:i])
        y.append(scaled_features[i, 3])  # Prediccion del precio de cierre (Close)

    X, y = np.array(X), np.array(y)

    """Separación en Conjunto de Entrenamiento y Prueba

    """

    # Divide en conjuntos de entrenamiento y prueba
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    """### Definición y Compilación del Modelo LSTM"""

    # Definición y Compilación del Modelo LSTM

    # Construir el modelo LSTM
    model = Sequential() # Se inicializa un modelo secuencial

    # Se añade una capa LSTM con 50 unidades, que devuelve secuencias para ser utilizadas en la siguiente capa LSTM
    # Se especifica la forma de entrada como (número de timesteps, número de características)
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))

    # Se añade una segunda capa LSTM con 50 unidades, sin devolver secuencias (última capa LSTM)
    model.add(LSTM(units=50))

    # Se añade una capa densa completamente conectada con una unidad de salida (predicción final)
    model.add(Dense(1))

    # Compilar el modelo
    # Se especifica el optimizador 'adam' y la función de pérdida 'mean_squared_error'
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    """Predicciones"""

    # Hacer predicciones
    predictions = model.predict(X_test)

    # Invertir la escala de las predicciones
    predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], scaled_features.shape[1]-1))), axis=1))[:,0]

    """Gráfica de las Predicciones y Valores Reales"""

    # Crear un DataFrame para las predicciones y los valores reales
    predictions_df = pd.DataFrame({
        'Fecha': bvn_df.index[-len(predictions):],
        'Predicciones': predictions,
        'Valores Reales': scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaled_features.shape[1]-1))), axis=1))[:,0]
    })

    # Graficar las predicciones y los valores reales
    fig5 = go.Figure()

    # Añadir la línea de Valores Reales
    fig5.add_trace(go.Scatter(
        x=predictions_df['Fecha'],
        y=predictions_df['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))

    # Añadir la línea de Predicciones
    fig5.add_trace(go.Scatter(
        x=predictions_df['Fecha'],
        y=predictions_df['Predicciones'],
        mode='lines',
        name='Predicciones',
        line=dict(color='brown')
    ))

    # Configurar el diseño de la figura
    fig5.update_layout(
        title='Predicciones vs Valores Reales del Precio de Cierre',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
)


    """### Evaluación del modelo"""

    # Calcular las métricas de evaluación
    mse = mean_squared_error(predictions_df['Valores Reales'], predictions_df['Predicciones'])
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(predictions_df['Valores Reales'], predictions_df['Predicciones'])

    return mse , rmse , mape, fig1, fig2, fig3, fig4, fig5


