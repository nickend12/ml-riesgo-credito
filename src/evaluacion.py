import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def evaluar_modelo(modelo, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "informes") -> dict:
    """
    Evalúa el modelo con el conjunto de prueba, calcula las métricas
    de negocio y genera la gráfica de la matriz de confusión.
    """
    print("--- INICIANDO FASE DE EVALUACIÓN Y MÉTRICAS ---")
    
    # 1. Realizar predicciones sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)
    
    # 2. Cálculo de métricas principales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    reporte = classification_report(y_test, y_pred)
    
    print(f"📊 Exactitud (Accuracy):   {accuracy:.4f}")
    print(f"🎯 Precisión (Precision):   {precision:.4f}")
    print(f"🔍 Sensibilidad (Recall):   {recall:.4f}")
    print(f"⚖️ Puntuación F1 (F1-Score): {f1:.4f}\n")
    print("Reporte de Clasificación detallado:")
    print(reporte)
    
    # 3. Generar y guardar la gráfica de la Matriz de Confusión
    os.makedirs(output_dir, exist_ok=True)
    ruta_imagen = os.path.join(output_dir, "matriz_confusion.png")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Cumple (0)', 'Mora (1)'],
        yticklabels=['Cumple (0)', 'Mora (1)']
    )
    plt.title('Matriz de Confusión - Riesgo de Crédito')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(ruta_imagen)
    plt.close()
    
    print(f"🖼️ Gráfica de matriz de confusión guardada en: '{ruta_imagen}'\n")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "reporte": reporte
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    
    from src.ingesta import cargar_datos_huggingface
    from src.preprocesamiento import preprocesar_datos
    from src.modelo import entrenar_modelo_random_forest
    
    # Prueba unitaria de la evaluación
    df_raw = cargar_datos_huggingface()
    X_train, X_test, y_train, y_test, _ = preprocesar_datos(df_raw)
    modelo = entrenar_modelo_random_forest(X_train, y_train)
    
    metricas = evaluar_modelo(modelo, X_test, y_test)