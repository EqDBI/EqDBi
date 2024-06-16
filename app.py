import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt

# Importar modelos
from ModelosPrediccion.LSTM import ejecutar_lstm
from ModelosPrediccion.MLP import ejecutar_mlp
from ModelosPrediccion.RBF_SVR import ejecutar_rbf_svr
from ModelosPrediccion.SVR import ejecutar_svr
import plotly.graph_objects as go  # Importa Plotly para la integración con Streamlit


modelos_disponibles = ["LSTM", "MLP", "RBF + SVR", "SVR"]
instrumento_financiero = ['BVN', 'FSM', 'SSCO', 'GOLD', 'AUY', 'GFI', 'HMY']
                          

@st.cache_data
def obtener_datos(instrumento_financiero, start, end):
    df = yf.download(instrumento_financiero, start=start, end=end)
    return df

st.title("Evaluación de Modelos de Predicción de Stock")

modelo_seleccionado = st.selectbox("Seleccione el modelo para la evaluación", modelos_disponibles)
instrumento_financiero= st.selectbox("Seleccione el instrumento financiero", instrumento_financiero)
fecha_inicio = st.date_input("Fecha de inicio", value=datetime(2020, 1, 1))
fecha_fin = st.date_input("Fecha de fin", value=datetime(2023, 1, 1))

if st.button("Evaluar"):
    fig6 = None
    if fecha_inicio < fecha_fin:
        df = obtener_datos(instrumento_financiero, fecha_inicio, fecha_fin)

        if not df.empty:
            st.subheader(f"Datos del stock de {instrumento_financiero} desde {fecha_inicio} hasta {fecha_fin}")
            st.write(df)
            st.line_chart(df['Close'])

            if modelo_seleccionado == "LSTM":
                mse , rmse , mape, fig1, fig2, fig3, fig4, fig5= ejecutar_lstm(df,instrumento_financiero, fecha_inicio, fecha_fin)

            elif modelo_seleccionado == "MLP":
                mse, rmse, mape, fig1, fig2, fig3, fig4, fig5 = ejecutar_mlp(df,instrumento_financiero, fecha_inicio, fecha_fin)

            elif modelo_seleccionado == "RBF + SVR":
                mse, rmse, mape, fig1, fig2, fig3, fig4, fig5 = ejecutar_rbf_svr(df,instrumento_financiero, fecha_inicio, fecha_fin)

            elif modelo_seleccionado == "SVR":
                mse , rmse , mape,fig1, fig2, fig3, fig4, fig5, fig6= ejecutar_svr(df,instrumento_financiero, fecha_inicio, fecha_fin)
                
            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.plotly_chart(fig3)
            st.plotly_chart(fig4)
            st.plotly_chart(fig5)
            if (fig6 is not None):
                st.plotly_chart(fig6)
            else:
                pass

            st.subheader("Métricas de Evaluación")
            st.write(f"Error Cuadrático Medio (MSE): {mse}")
            st.write(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
            st.write(f"Error Absoluto Porcentual Promedio (MAPE): {mape}")
        else:
            st.error("No se encontraron datos para el rango de fechas seleccionado.")
    else:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")