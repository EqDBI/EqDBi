import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import yfinance as yf

def preprocessing(df):
    # Calcular la media móvil exponencial
    df['EMA'] = df['Open'].ewm(span=20, adjust=False).mean()

    # Graficar la variación en el tiempo del precio de apertura y la media móvil exponencial
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df.index, y=df['Open'], mode='lines', name='Open Price'))
    fig1.add_trace(go.Scatter(x=df.index, y=df['EMA'], mode='lines', name='EMA (20 días)', line=dict(color='red')))
    fig1.update_layout(
        title='Variación en el tiempo del precio de apertura (Open) con EMA',
        xaxis_title='Fecha',
        yaxis_title='Precio de apertura',
        legend_title='Legend',
        template='plotly_white',
        xaxis=dict(rangeslider=dict(visible=True))
    )

    # Variacion del volumen (Volume) en el tiempo (escala = e8)
    fig_volume = plt.figure()
    sns.lineplot(data=df, x=df.index, y="Volume")
    plt.title("Volume Over Time")

    # Calcular los retornos diarios
    df['Returns'] = df['Close'].pct_change()

    # Graficar la distribución de los retornos
    fig2 = ff.create_distplot(
        [df['Returns'].dropna()], 
        group_labels=['Returns'], 
        bin_size=0.01, 
        show_curve=True, 
        colors=['blue']
    )
    fig2.update_layout(
        title='Distribución de Retornos Diarios',
        xaxis_title='Retornos',
        yaxis_title='Frecuencia',
        template='plotly_white'
    )

    # Descomposición de la serie temporal
    decomposition = seasonal_decompose(df['Close'].dropna(), model='multiplicative', period=365)
    fig_decomposition = decomposition.plot()

    # Añadir nuevas columnas basadas en la descripción
    df['Precio Anterior'] = df['Close'].shift(1)
    df['Precio Máximo Anterior'] = df['High'].shift(1)
    df['Precio Mínimo Anterior'] = df['Low'].shift(1)
    df['Precio Apertura Anterior'] = df['Open'].shift(1)
    df['PM_10'] = df['Close'].rolling(window=10).mean()

    # Calcular bandas de Bollinger
    df['Middle Band Bollinger'] = df['Close'].rolling(window=20).mean()
    df['Upper Band Bollinger'] = df['Middle Band Bollinger'] + 1.96 * df['Close'].rolling(window=20).std()
    df['Lower Band Bollinger'] = df['Middle Band Bollinger'] - 1.96 * df['Close'].rolling(window=20).std()

    # Gráfico: Precio de Cierre con Bandas de Bollinger
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Precio de Cierre'))
    fig3.add_trace(go.Scatter(x=df.index, y=df['Upper Band Bollinger'], mode='lines', name='Banda Superior Bollinger', line=dict(color='green')))
    fig3.add_trace(go.Scatter(x=df.index, y=df['Lower Band Bollinger'], mode='lines', name='Banda Inferior Bollinger', line=dict(color='red')))
    fig3.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=df['Upper Band Bollinger'].tolist() + df['Lower Band Bollinger'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    fig3.update_layout(
        title='Precio de Cierre con Bandas de Bollinger',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    # Calcular precio medio
    df['Precio Medio'] = (df['High'] + df['Low'] + df['Close']) / 3

    # Precio de la plata
    silver_df = yf.download('SI=F', start=df.index.min(), end=df.index.max())
    df['Precio Plata'] = silver_df['Close']

    # Agregar la columna del precio del día siguiente
    df['Precio Siguiente'] = df['Close'].shift(-1)

    # Gráfico: Comparación del Precio de Cierre con el Precio de la Plata
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name=f'Precio de Cierre'))
    fig4.add_trace(go.Scatter(x=df.index, y=df['Precio Plata'], mode='lines', name='Precio de la Plata', line=dict(color='silver')))
    fig4.update_layout(
        title='Comparación del Precio de Cierre con el Precio de la Plata',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white'
    )

    # Eliminar filas con valores NaN generados por los cálculos de rolling y shift
    df.dropna(inplace=True)

    return df, fig1, fig2, fig3, fig4, fig_volume