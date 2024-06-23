# Basicos
import pandas as pd # Manipulación y análisis de datos
import numpy as np # Operaciones numéricas y algebra lineal

# Preprocesamiento

from sklearn.preprocessing import MinMaxScaler # Escalado de características a un rango específico

# Evaluacion
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error # Métricas de evaluación


from sklearn.svm import SVR # Soporte vectorial para regresión

# Modelo Híbrido
from sklearn.kernel_approximation import RBFSampler # Aproximación de kernel de base radial (RBF)
from sklearn.pipeline import make_pipeline # Creación de pipelines para flujos de trabajo de ML
from ModelosPrediccion.preprocessing import preprocessing

import plotly.graph_objects as go # Visualizaciones interactivas


"""# COMPAÑIA DE MINAS BUENAVENURA SAA (BVN)

## Extracción de datos
"""
def ejecutar_rbf_svr(df,instrumento_financiero, fecha_inicio, fecha_fin):
    bvn_df, fig1, fig2, fig3, fig4 = preprocessing(df,instrumento_financiero, fecha_inicio, fecha_fin)


    """## MODELO: MODELO HÍBRIDO (RBF+SVR)

    """

    # Seleccionar las columnas que quieres usar como input para el modelo
    features = bvn_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Precio Anterior',
                    'Precio Máximo Anterior', 'Precio Mínimo Anterior', 'Precio Apertura Anterior',
                    'PM_10', 'Middle Band Bollinger', 'Upper Band Bollinger',
                    'Lower Band Bollinger', 'Precio Medio', 'Precio Plata']].values

    # Escalar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)

    # Definir la longitud de la secuencia (número de timesteps)
    timesteps = 60

    # Prepara los datos para el modelo
    X = []
    y = []

    for i in range(timesteps, len(scaled_features)):
        X.append(scaled_features[i-timesteps:i])
        y.append(scaled_features[i, 3])  # Supongamos que quieres predecir el precio de cierre (Close)

    X, y = np.array(X), np.array(y)

    # Divide en conjuntos de entrenamiento y prueba
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    """### Definicion del modelo"""

    # Ajustar gamma y aumentar el número de componentes
    # Se crea un transformador RBF Sampler con un valor específico de gamma y un número específico de componentes
    rbf_feature = RBFSampler(gamma=0.01, n_components=500, random_state=1)
    # Se crea un modelo de Support Vector Regression (SVR) con kernel lineal
    svm = SVR(kernel='linear')
    # Se combina el RBF Sampler y el SVR en un pipeline para formar el modelo híbrido
    model_rbf = make_pipeline(rbf_feature, svm)

    # Entrenar el modelo
    # Se ajusta el pipeline a los datos de entrenamiento, reestructurando X_train para que tenga el formato adecuado
    model_rbf.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    # Hacer predicciones
    # Se realizan predicciones sobre el conjunto de prueba, reestructurando X_test para que tenga el formato adecuado
    predictions_rbf = model_rbf.predict(X_test.reshape(X_test.shape[0], -1))

    # Invertir la escala de las predicciones y de y_test
    # Se concatenan las predicciones con ceros para ajustarlas a la estructura original antes del escalado y se invierte la escala
    predictions_rbf = scaler.inverse_transform(
        np.concatenate((predictions_rbf.reshape(-1, 1),
                        np.zeros((predictions_rbf.shape[0], scaled_features.shape[1] - 1))), axis=1))[:, 0]
    # Se hace lo mismo para y_test
    y_test_rbf = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1),
                        np.zeros((y_test.shape[0], scaled_features.shape[1] - 1))), axis=1))[:, 0]

    # Crear un DataFrame para las predicciones y los valores reales en el conjunto de prueba
    test_dates = bvn_df.index[-len(predictions_rbf):]
    predictions_df_rbf = pd.DataFrame({
        'Fecha': test_dates,
        'Predicciones RBF': predictions_rbf,
        'Valores Reales': y_test_rbf
    })

    # Graficar las predicciones y los valores reales del conjunto de prueba para RBF
    fig5 = go.Figure()

    # Añadir la línea de Valores Reales
    fig5.add_trace(go.Scatter(
        x=predictions_df_rbf['Fecha'],
        y=predictions_df_rbf['Valores Reales'],
        mode='lines',
        name='Valores Reales'
    ))

    # Añadir la línea de Predicciones RBF
    fig5.add_trace(go.Scatter(
        x=predictions_df_rbf['Fecha'],
        y=predictions_df_rbf['Predicciones RBF'],
        mode='lines',
        name='Predicciones RBF',
        line=dict(color='red')
    ))

    # Configurar el diseño de la figura
    fig5.update_layout(
        title='Predicciones RBF vs Valores Reales del Precio de Cierre (Conjunto de Prueba)',
        xaxis_title='Fecha',
        yaxis_title='Precio de Cierre',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    """### Evaluación del modelo"""

    # Calcular las métricas de evaluación
    mse = mean_squared_error(y_test_rbf, predictions_rbf)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_rbf, predictions_rbf)


    return mse, rmse, mape, fig1, fig2, fig3, fig4, fig5