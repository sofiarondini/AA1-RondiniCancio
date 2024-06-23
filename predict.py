# Librerias
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from funciones import *
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import save_model
import os
import warnings
warnings.filterwarnings("ignore")



### IMPORTAR LOS DATOS FUENTE DEL ARCHIVO CSV COMO DATAFRAME DE PANDAS
file_path = "https://raw.githubusercontent.com/sofiarondini/AA1-RondiniCancio/main/weatherAUS%20(1).csv"
df= leer_csv_desde_url(file_path, separador=',')
model_dir  = 'C:/AA1/TPIntegrador/AA1-TUIA-Rondini-Cancio/'

# PREPARACION DATOS FUENTE
df = preparacion_fuente(df)

# MODELO PARA CLASIFICACIÓN
X_train, y_train = dividir_y_manipular(df, 'clasificacion')

# Aplicamos SMOTE para oversamplear la clase monoritaria
X_resampled, y_resampled = SMOTE(sampling_strategy='minority').fit_resample(X_train, y_train)

pipeline_clas = Pipeline([
    ('scaler', StandardScaler()),
    ('nn_classifier', NeuralClassifier(build_fn=nn_clas))
])

pipeline_clas.fit(X_resampled, y_resampled)
model_clas = pipeline_clas.named_steps['nn_classifier'].get_model()
save_model(model_clas, os.path.join(model_dir, 'keras_model_clas.h5'))
pipeline_clas.named_steps['nn_classifier'].set_model(None)
joblib.dump(pipeline_clas, os.path.join(model_dir, 'pipeline_clas.joblib'))

# MODELO PARA REGRESIÓN
X_train, y_train = dividir_y_manipular(df, 'regresion')
pipeline_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('nn_regressor', NeuralRegressor(build_fn=nn_reg))
])
pipeline_reg.fit(X_train, y_train)
model_reg = pipeline_reg.named_steps['nn_regressor'].get_model()
save_model(model_reg, os.path.join(model_dir, 'keras_model_reg.h5'))
pipeline_reg.named_steps['nn_regressor'].set_model(None)
joblib.dump(pipeline_reg, os.path.join(model_dir, 'pipeline_reg.joblib'))

