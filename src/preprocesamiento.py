import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocesar_datos(df: pd.DataFrame, target_col: str = 'default.payment.next.month'):
    """
    Limpia el DataFrame, separa variables predictoras y objetivo, 
    realiza la división train/test y escala las variables numéricas
    previniendo la fuga de información (data leakage).
    """
    print("--- INICIANDO FASE DE PREPROCESAMIENTO ---")
    print(f"Dimensiones iniciales: {df.shape[0]} filas, {df.shape[1]} columnas.")
    
    # 1. Limpieza básica: Eliminación de nulos y duplicados
    nulos_iniciales = df.isnull().sum().sum()
    duplicados_iniciales = df.duplicated().sum()
    print(f"Valores nulos detectados: {nulos_iniciales}")
    print(f"Filas duplicadas detectadas: {duplicados_iniciales}")
    
    df_limpio = df.drop_duplicates().copy()
    if nulos_iniciales > 0:
        df_limpio = df_limpio.dropna()
        
    print(f"Dimensiones tras limpieza: {df_limpio.shape[0]} filas x {df_limpio.shape[1]} columnas.")
    
    # Confirmar el nombre exacto de la columna objetivo
    if target_col not in df_limpio.columns:
        posibles = [col for col in df_limpio.columns if 'default' in col.lower()]
        if posibles:
            target_col = posibles[0]
            
    print(f"Variable objetivo identificada: '{target_col}'")
    
    # 2. Separación de características (X) y variable objetivo (y)
    X = df_limpio.drop(columns=[target_col])
    y = df_limpio[target_col]
    
    # Si la columna 'ID' o 'id' existe, se elimina por no ser un atributo predictivo
    cols_id = [c for c in X.columns if c.lower() in ['id', 'unnamed: 0']]
    if cols_id:
        X = X.drop(columns=cols_id)
        print(f"Columna de identificador eliminada: {cols_id}")
        
    # 3. División Train/Test (80% entrenamiento, 20% prueba con estratificación)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Conjunto de Entrenamiento (Train): {X_train.shape[0]} muestras")
    print(f"Conjunto de Prueba (Test): {X_test.shape[0]} muestras")
    
    # 4. Escalado de variables previniendo Data Leakage
    # IMPORTANTE: fit_transform se ejecuta ÚNICAMENTE sobre X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # transform aplica las medias y desviaciones aprendidas de Train sobre X_test
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir los arreglos a DataFrames para mantener los nombres de las columnas
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print("✅ Preprocesamiento finalizado con éxito (sin data leakage).\n")
    
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, scaler


if __name__ == "__main__":
    from src.ingesta import cargar_datos_huggingface
    
    # Prueba unitaria del preprocesamiento
    df_raw = cargar_datos_huggingface()
    X_tr, X_te, y_tr, y_te, _ = preprocesar_datos(df_raw)
    print("Muestra de datos de entrenamiento escalados:")
    print(X_tr.head(3))