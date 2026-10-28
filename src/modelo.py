import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def entrenar_modelo_random_forest(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    n_estimators: int = 100, 
    max_depth: int = 10, 
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Configura y entrena un modelo Random Forest Classifier
    para la predicción de riesgo de impago de crédito.
    """
    print("--- INICIANDO ENTRENAMIENTO DEL MODELO ---")
    print("Algoritmo seleccionado: Random Forest Classifier (Supervisado)")
    print(f"Hiperparámetros: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}")
    
    # Inicializar el modelo con semilla aleatoria fija
    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Entrenamiento con el conjunto de entrenamiento (X_train, y_train)
    modelo.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente con el conjunto de entrenamiento.\n")
    
    return modelo


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Permitir la importación relativa ejecutando el archivo directamente
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from src.ingesta import cargar_datos_huggingface
    from src.preprocesamiento import preprocesar_datos
    
    # Prueba unitaria del entrenamiento
    df_raw = cargar_datos_huggingface()
    X_train, X_test, y_train, y_test, scaler = preprocesar_datos(df_raw)
    
    modelo = entrenar_modelo_random_forest(X_train, y_train)
    print(f"Número de características procesadas: {modelo.n_features_in_}")