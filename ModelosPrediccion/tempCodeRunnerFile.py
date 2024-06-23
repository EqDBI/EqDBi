import yfinance as yf
import pandas as pd

# Definir el símbolo del instrumento financiero (por ejemplo, 'AAPL' para Apple)
symbol = 'AAPL'

# Obtener los datos del instrumento financiero hasta el día de hoy
data = yf.download(symbol, progress=False)

# Imprimir la cola del dataset extraído
print(data.tail())
