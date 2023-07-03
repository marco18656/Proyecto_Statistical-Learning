import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.utils import shuffle
import joblib
import os
import sys

def handle_missing_values(data):
    # Imputar valores faltantes numéricos con la mediana
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    
    # Imputar valores faltantes categóricos con la moda
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])
    
    return data

def handle_outliers(data):
    # Aplicar el método de detección y corrección de outliers en las columnas numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        # Detectar y corregir outliers (por ejemplo, utilizando el rango intercuartílico)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    return data

def preprocess_data(data):
    # Manejar valores faltantes
    data = handle_missing_values(data)
    
    # Manejar outliers
    data = handle_outliers(data)
    
    # Codificar variables categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names(categorical_cols))
    data.drop(categorical_cols, axis=1, inplace=True)
    data = pd.concat([data, encoded_df], axis=1)
    
    # Estandarizar variables numéricas
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

def train_model(data_file):
    # Cargar el archivo CSV
    data = pd.read_csv(data_file)
    
    # Preprocesar los datos
    data = preprocess_data(data)
    
    # Separar las características de la variable objetivo
    X = data.drop('unique_leads', axis=1)
    y = data['unique_leads']
    
    # Verificar la distribución de clases en y
    if len(y.unique()) < 2:
        print('Error: Solo hay una clase presente en los datos. Verifica la distribución de clases.')
        sys.exit(1)
    
    # Dividir el conjunto de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
    # Definir las transformaciones para las características numéricas y categóricas
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combinar las transformaciones en un preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ('numeric', numeric_transformer, X_train.select_dtypes(include=['float64', 'int64']).columns),
    ])
    
    # Crear el pipeline completo con el preprocesador y el clasificador
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)
    
    # Generar predicciones en el conjunto de entrenamiento
    y_train_pred = pipeline.predict(X_train)
    
    # Calcular las métricas de entrenamiento
    accuracy = accuracy_score(y_train, y_train_pred)
    specificity = recall_score(y_train, y_train_pred, pos_label=0)
    sensitivity = recall_score(y_train, y_train_pred, pos_label=1)
    
    # Verificar la distribución de clases en y_train
    if len(y_train.unique()) < 2:
        print('Error: Solo hay una clase presente en los datos de entrenamiento. Verifica la distribución de clases.')
        sys.exit(1)
    
    # Calcular el puntaje ROC AUC solo si hay más de una clase presente en y_train
    if len(y_train.unique()) > 1:
        roc_auc = roc_auc_score(y_train, y_train_pred)
    else:
        roc_auc = 0.0
    
    # Guardar los resultados en un archivo de texto
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Crear el nombre del archivo de resultados
    result_file = os.path.join(script_dir, f'results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt')

    # Guardar los resultados en el archivo de resultados
    with open(result_file, 'w') as f:
        f.write(f'Fecha y hora de ejecución: {pd.Timestamp.now()}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Specificity: {specificity}\n')
        f.write(f'Sensitivity: {sensitivity}\n')
        f.write(f'ROC-AUC: {roc_auc}\n')

    print(f'Entrenamiento completo. Resultados guardados en {result_file}')
    joblib.dump(pipeline, 'model.pkl')
    print('Modelo guardado como model.pkl')

def predict(data_file, output_file):
    # Cargar el archivo CSV de datos a predecir
    data = pd.read_csv(data_file)
    
    # Preprocesar los datos
    data = preprocess_data(data)
    
    # Cargar el modelo entrenado
    pipeline = joblib.load('model.pkl')
    
    # Predecir las etiquetas de los datos
    y_pred = pipeline.predict(data)
    
    # Guardar las predicciones en un archivo CSV
    data['prediction'] = y_pred
    data.to_csv(output_file, index=False)
    
    print(f'Predicciones generadas y guardadas en {output_file}')

# Ejecución del script desde la consola
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Debe especificar al menos una operación: train o predict')
        sys.exit(1)
    
    operation = sys.argv[1]
    
    if operation == 'train':
        if len(sys.argv) < 3:
            print('Debe especificar el archivo de datos para entrenar')
            sys.exit(1)
        
        data_file = sys.argv[2]
        train_model(data_file)
        
    elif operation == 'predict':
        if len(sys.argv) < 4:
            print('Debe especificar el archivo de datos a predecir y el archivo de salida')
            sys.exit(1)
        
        data_file = sys.argv[2]
        output_file = sys.argv[3]
        predict(data_file, output_file)
        
    else:
        print('Operación no válida. Las opciones son train o predict')
        sys.exit(1)
