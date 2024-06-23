# Librerias
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, f1_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import BaseEstimator, RegressorMixin
import joblib
from joblib import Parallel, delayed
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.regularizers import l2
from keras.losses import MeanSquaredError, BinaryCrossentropy
from keras.metrics import MeanSquaredError as MSE
import random


# LEER DATASET
def leer_csv_desde_url(url, separador=','):
    df = pd.read_csv(url, sep=separador, engine='python')
    return df

# FUNCIONES DE PREPROCESAMIENTO DE DATOS

def preprocesar_datos(df):
    df['Date'] = pd.to_datetime(df['Date'])  # Convertir columna Date a datetime

    # Filtrar por ubicaciones específicas
    ubicaciones = ['Adelaide', 'Canberra', 'Cobar', 'Dartmoor', 'Melbourne',
                   'MelbourneAirport', 'MountGambier', 'Sydney', 'SydneyAirport']
    df_filtrado = df[df['Location'].isin(ubicaciones)]

    # Eliminar columnas no deseadas
    if 'Unnamed: 0' in df_filtrado.columns:
        df_filtrado.drop(columns='Unnamed: 0', inplace=True)
    if 'Location' in df_filtrado.columns:
        df_filtrado.drop(columns='Location', inplace=True)

    df_filtrado.sort_values(by='Date', inplace=True)

    return df_filtrado


def aplicar_transformaciones(df):
    transformaciones = {
        'MinTemp': lambda x: min(max(x, -20), 50),
        'MaxTemp': lambda x: min(max(x, -20), 50),
        'Rainfall': lambda x: max(x, 0),
        'Evaporation': lambda x: max(x, 0),
        'Sunshine': lambda x: min(max(x, 0), 24),
        'WindGustSpeed': lambda x: max(x, 0),
        'WindSpeed9am': lambda x: max(x, 0),
        'WindSpeed3pm': lambda x: max(x, 0),
        'Temp9am': lambda x: min(max(x, -20), 50),
        'Temp3pm': lambda x: min(max(x, -20), 50),
    }

    for col, transformation in transformaciones.items():
        if col in df.columns:
            df[col] = df[col].apply(transformation)

    # Calcular nuevas características si las columnas existen
    if all(col in df.columns for col in ['Temp3pm', 'Temp9am']):
        df['Temperature'] = df['Temp3pm'] - df['Temp9am']
    if all(col in df.columns for col in ['Pressure3pm', 'Pressure9am']):
        df['Pressure'] = df['Pressure3pm'] - df['Pressure9am']
    if all(col in df.columns for col in ['Humidity3pm', 'Humidity9am']):
        df['Humidity'] = df['Humidity3pm'] - df['Humidity9am']
    if all(col in df.columns for col in ['Cloud3pm', 'Cloud9am']):
        df['Cloud'] = df['Cloud3pm'] - df['Cloud9am']

    # Eliminar columnas temporales si existen
    col_a_eliminar = ['Temp9am', 'Temp3pm', 'Pressure9am', 'Pressure3pm', 'Humidity9am', 'Humidity3pm', 'Cloud9am', 'Cloud3pm']
    df = df.drop(columns=[col for col in col_a_eliminar if col in df.columns])
    return df


def codificar_categoricos(df):
    categoricas = df.select_dtypes(include='object').columns
    for col in categoricas:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df

def procesar_datos_faltantes(df):
    num_var = [variable for variable in df.columns if df[variable].dtypes in ['int64', 'float64']]
    cat_var = [variable for variable in df.columns if df[variable].dtypes == 'object']

    medianas = df[num_var].median()
    modas = df[cat_var].mode().iloc[0] if not df[cat_var].empty else pd.Series([], index=cat_var)

    for col in df.columns:
        if df[col].isnull().sum() < 5:
            df = df.dropna(subset=[col])

    def cambiar_mediana(df, columna_objetivo, medianas):
        df[columna_objetivo].fillna(medianas[columna_objetivo], inplace=True)

    def cambiar_moda(df, columna_objetivo, modas):
        df[columna_objetivo].fillna(modas[columna_objetivo], inplace=True)

    for col in num_var:
        cambiar_mediana(df, col, medianas)

    for col in cat_var:
        cambiar_moda(df, col, modas)

    return df


def dividir_y_manipular(df,problema='regresion'):
    if problema == 'regresion':
        # Separar X e y
        df = df.drop(columns='Date') 
        X = df.drop(columns=['RainfallTomorrow', 'RainTomorrow'])
        y = df['RainfallTomorrow']
        #Conjuntos de datos para entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
     
    elif problema == 'clasificacion':
        # Separar X e y
        df = df.drop(columns='Date') 
        X = df.drop(columns=['RainfallTomorrow', 'RainTomorrow'])
        y = df['RainTomorrow']
        #Conjuntos de datos para entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        # y_train = y_train.astype(float)

    return X_train, y_train


def nn_clas(): 
    # 
    hp = {
        'num_layers': 2,
        'n_units_layer_0': 81,
        'n_units_layer_1': 44,
        'epochs': 28,
        'batch_size': 473
    }

    # Crear el modelo
    model = Sequential()
    model.add(Dense(hp['n_units_layer_0'], activation='relu'))
    model.add(Dense(hp['n_units_layer_1'], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Recall'])
    return model

def nn_reg(): 
    # Best parámetros encontrados: {'num_layers': 1, 'n_units_layer_0': 44, 'epochs': 89, 'batch_size': 260}
    hp = {
        'num_layers': 1,
        'n_units_layer_0': 44,
        'epochs': 89,
        'batch_size': 260
    }

    # Crear el modelo
    model = Sequential()
    model.add(Dense(hp['n_units_layer_0'], activation='relu'))
    model.add(Dense(1, activation='linear'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


# PREPARACION DATOS FUENTE
def preparacion_fuente(df):
    df1 = preprocesar_datos(df)
    df2 = aplicar_transformaciones(df1)
    df3 = codificar_categoricos(df2)
    df4 = procesar_datos_faltantes(df3)
    return df4


# PREPARACION DATOS PREDICCION
def preparacion_test(df):
    df1 = preprocesar_datos(df)
    df2 = aplicar_transformaciones(df1)
    df3 = codificar_categoricos(df2)
    df3.drop(columns=['Date','RainfallTomorrow', 'RainTomorrow'], inplace=True)
    return df3


class NeuralClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, build_fn):
        self.build_fn = build_fn
    
    def fit(self, X, y):
        self.model_ = self.build_fn()
        self.model_.fit(X, y, epochs=69, batch_size=2048, verbose=0)
        return self
    
    def predict(self, X):
        y_pred = self.model_.predict(X)
        return (y_pred > 0.4).astype("int32")

    def predict_proba(self, X):
        return self.model_.predict(X)
    
    def get_model(self):
        return self.model_
    
    def set_model(self, model):
        self.model_ = model
    
class NeuralRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, build_fn, epochs=74, batch_size=444, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
    
    def fit(self, X, y):
        self.model_ = self.build_fn()
        self.model_.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self
    
    def predict(self, X):
        return self.model_.predict(X).flatten()

    def predict_proba(self, X):
        return self.model_.predict(X)
    
    def get_model(self):
        return self.model_
    
    def set_model(self, model):
        self.model_ = model

def load_pipeline_and_model(root_path, pipeline_name, keras_model_name, step_name):
    pipeline_path = f'{root_path}{pipeline_name}'
    keras_model_path = f'{root_path}{keras_model_name}'
    pipeline = joblib.load(pipeline_path)
    model = load_model(keras_model_path)
    pipeline.named_steps[step_name].set_model(model)
    return pipeline