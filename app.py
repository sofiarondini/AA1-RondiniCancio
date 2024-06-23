# app.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
import joblib
from funciones import *


# Título
st.title('Predicción de lluvia en Australia')

# Carga de pipelines
model_path = 'C:/AA1/TPIntegrador/AA1-TUIA-Rondini-Cancio/'
pipeline_clas = load_pipeline_and_model(model_path, 'pipeline_clas.joblib', 'keras_model_clas.h5', 'nn_classifier')
pipeline_reg = load_pipeline_and_model(model_path, 'pipeline_reg.joblib', 'keras_model_reg.h5', 'nn_regressor')

# Controles deslizantes

st.header('Valores de las características para predecir:')
direcciones_viento = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
date = st.date_input('Date')
location = st.selectbox('Location', ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne', 'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport'])
rain_today = st.selectbox('RainToday', ['No', 'Yes'])

# Crear columnas para los sliders
col1, col2, col3 = st.columns(3)
with col1:
    min_temp = st.slider('Temperatura mínima', -20.0, 50.0, 20.0)
    max_temp = st.slider('Temperatura máxima', -20.0, 50.0, 30.0)
    rainfall = st.slider('Precipitacion', 0.0, 100.0, 10.0)
    evaporation = st.slider('Evaporación', 0.0, 20.0, 5.0)
    sunshine = st.slider('Horas de sol', 0.0, 24.0, 12.0)
    wind_gust_dir = st.select_slider('Dirección  Rafaga Viento:', options=direcciones_viento, value=direcciones_viento[0])
    wind_gust_speed = st.slider('Velocidad máxima de ráfaga de viento', 0.0, 150.0, 50.0)

with col2:
    wind_dir_9am = st.select_slider('Direccion Viento a las 9AM:', options=direcciones_viento, value=direcciones_viento[0])
    wind_dir_3pm = st.select_slider('Direccion Viento a las 3PM', options=direcciones_viento, value=direcciones_viento[0])
    wind_speed_9am = st.slider('Velocidad del viento a las 9am', 0.0, 200.0, 10.0)
    wind_speed_3pm = st.slider('Velocidad del viento a las 3pm', 0.0, 200.0, 15.0)
    temp_9am = st.slider('Temperatura a las 9AM', -20.0, 50.0, 20.0)
    temp_3pm = st.slider('Temperatura a las 3PM', -20.0, 50.0, 20.0)

with col3:
    pressure_9am = st.slider('Presión a las 9AM', 900.0, 1100.0, 1000.0)
    pressure_3pm = st.slider('Presión a las 3PM', 900.0, 1100.0, 1000.0)
    humidity_9am = st.slider('Humedad a las 9AM', 0.0, 100.0, 40.0)
    humidity_3pm = st.slider('Humedad a las 3PM', 0.0, 100.0, 40.0)
    cloud_3pm = st.slider('Nubosidad a las 9AM', 0.0, 9.0, 5.0)
    cloud_9am = st.slider('Nubosidad a las 3PM', 0.0, 9.0, 5.0)


# Datos ingresados para la predicción
df_pred = pd.DataFrame({
    'Date': [date],
    'Location': [location],
    'MinTemp': [min_temp],
    'MaxTemp': [max_temp],
    'Rainfall': [rainfall],
    'Evaporation': [evaporation],
    'Sunshine': [sunshine],
    'WindGustDir': [wind_gust_dir],
    'WindGustSpeed': [wind_gust_speed],
    'WindDir9am': [wind_dir_9am],
    'WindDir3pm': [wind_dir_3pm],
    'WindSpeed9am': [wind_speed_9am],
    'WindSpeed3pm': [wind_speed_3pm],
    'Humidity9am': [humidity_9am],
    'Humidity3pm': [humidity_3pm],
    'Pressure9am': [pressure_9am],
    'Pressure3pm': [pressure_3pm],
    'Cloud9am': [cloud_9am],
    'Cloud3pm': [cloud_3pm],
    'Temp9am': [temp_9am],
    'Temp3pm': [temp_3pm],
    'RainToday': [rain_today],
    'RainTomorrow': [1],
    'RainfallTomorrow': [1]
})

# Crea el botón de "Predecir" y realiza la prediccion

if st.button('Predecir'):
    df_pred = preparacion_test(df_pred)
    prediccion_clas = pipeline_clas.predict(df_pred)
    prediccion_reg = pipeline_reg.predict(df_pred)

    # Ajuste de la predicción de RainfallTomorrow
    rainfall_tomorrow_prediction = int(max(0, round(prediccion_reg[0])) if prediccion_clas[0] == 1 else 0)

    st.sidebar.header('Valores de las predicciones:')
    st.sidebar.write('RainTomorrow')
    prediction_text = 'Yes' if prediccion_clas[0] == 1 else 'No'
    st.sidebar.write(prediction_text)

    st.sidebar.write('RainfallTomorrow [mm]')
    st.sidebar.text(rainfall_tomorrow_prediction)
