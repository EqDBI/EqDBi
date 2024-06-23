import streamlit as st
from datetime import datetime
import yfinance as yf

# Importar modelos
from ModelosPrediccion.LSTM import ejecutar_lstm
from ModelosPrediccion.MLP import ejecutar_mlp
from ModelosPrediccion.RBF_SVR import ejecutar_rbf_svr
from ModelosPrediccion.SVR import ejecutar_svr

# Importar preprocesamiento
from ModelosPrediccion.preprocessing import preprocessing

modelos_disponibles = ["LSTM", "MLP", "RBF + SVR", "SVR"]
instrumentos_financieros = ['BVN', 'FSM', 'SSCO', 'GOLD', 'AUY', 'GFI', 'HMY']

@st.cache_data
def obtener_datos(instrumento, start, end):
    df = yf.download(instrumento, start=start, end=end)
    return df

st.title("Evaluación de Modelos de Predicción de Stock")

modelo_seleccionado = st.selectbox("Seleccione el modelo para la evaluación", modelos_disponibles)
instrumento_seleccionado = st.selectbox("Seleccione el instrumento financiero", instrumentos_financieros)
fecha_inicio = st.date_input("Fecha de inicio", value=datetime(2020, 1, 1))
fecha_fin = st.date_input("Fecha de fin", value=datetime.today())
if st.button("Evaluar"):
    if fecha_inicio < fecha_fin:
        df = obtener_datos(instrumento_seleccionado, fecha_inicio, fecha_fin)

        if not df.empty:
            st.subheader(f"Datos del stock de {instrumento_seleccionado} desde {fecha_inicio} hasta {fecha_fin}")
            st.write(df)
            st.line_chart(df['Close'])

            # Preprocesamiento de datos
            df_preprocessed, fig1, fig2, fig3, fig4, fig5 = preprocessing(df)

            st.plotly_chart(fig1)
            st.plotly_chart(fig2)
            st.plotly_chart(fig3)
            st.plotly_chart(fig4)
            st.plotly_chart(fig5)


            if modelo_seleccionado == "LSTM":
                mse, rmse, mape, fig_pred = ejecutar_lstm(df_preprocessed)
            elif modelo_seleccionado == "MLP":
                mse, rmse, mape, fig_pred = ejecutar_mlp(df_preprocessed)

            elif modelo_seleccionado == "RBF + SVR":
                mse, rmse, mape, fig_pred = ejecutar_rbf_svr(df_preprocessed)

            elif modelo_seleccionado == "SVR":
                mse, rmse, mape, fig_pred = ejecutar_svr(df_preprocessed)

            st.plotly_chart(fig_pred)

                # Imprimir las métricas
            st.subheader("Métricas de Evaluación")
            st.write(f"Error Cuadrático Medio (MSE): {mse}")
            st.write(f"Raíz del Error Cuadrático Medio (RMSE): {rmse}")
            st.write(f"Error Absoluto Porcentual Promedio (MAPE): {mape}")

        else:
            st.error("No se encontraron datos para el rango de fechas seleccionado.")
    else:
        st.error("La fecha de inicio debe ser anterior a la fecha de fin.")
