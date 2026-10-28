import os
from datetime import datetime
from src.ingesta import cargar_datos_huggingface
from src.preprocesamiento import preprocesar_datos
from src.modelo import entrenar_modelo_random_forest
from src.evaluacion import evaluar_modelo


def generar_informe_hallazgos(metricas: dict, ruta_salida: str = "informes/hallazgos.md"):
    """
    Genera automáticamente el informe de hallazgos en formato Markdown
    inyectando las métricas calculadas en tiempo real.
    """
    contenido_md = f"""# 📊 Informe de Hallazgos y Análisis del Modelo de Riesgo de Crédito

> **Nota:** Este informe fue generado **automáticamente por el pipeline de ML** el {datetime.now().strftime('%Y-%m-%d a las %H:%M:%S')}.

## 1. Resumen Ejecutivo
Este documento analiza el desempeño del modelo **Random Forest Classifier** entrenado sobre el dataset `imodels/credit-card`. El objetivo es optimizar la toma de decisiones financieras en la aprobación de crédito, equilibrando la mitigación del riesgo con el crecimiento de la cartera.

---

## 2. Métricas de Evaluación Obtenidas (Conjunto de Prueba)

| Métrica | Valor Calculado | Interpretación de Negocio |
| :--- | :--- | :--- |
| **Accuracy (Exactitud)** | **{metricas['accuracy']*100:.2f}%** | Clasificación correcta general (cumplidos y morosos). |
| **Precision (Precisión)** | **{metricas['precision']*100:.2f}%** | De las alarmas de mora emitidas, este porcentaje realmente cayó en mora. |
| **Recall (Sensibilidad)** | **{metricas['recall']*100:.2f}%** | Porcentaje total de clientes en mora capturados por el modelo. |
| **F1-Score** | **{metricas['f1']*100:.2f}%** | Balance armónico entre Precisión y Sensibilidad. |

---

## 3. Matriz de Confusión y Análisis del Error Financiero

### A. Falsos Negativos (FN) — *El Riesgo Crítico*
* **Definición:** El modelo clasificó al cliente como **Cumplido (0)**, pero en la realidad **cayó en mora (1)**.
* **Impacto Financiero:** **Pérdida Directa de Capital.** Se otorga crédito a un cliente impagador, generando gastos de cobranza y cartera castigada.

### B. Falsos Positivos (FP) — *El Costo de Oportunidad*
* **Definición:** El modelo clasificó al cliente como **Moroso (1)**, pero el cliente **pagaba a tiempo (0)**.
* **Impacto Financiero:** **Pérdida de Ingresos.** Se rechaza a un cliente rentable, perdiendo cobros de intereses.

> **Conclusión:** En la banca, un **Falso Negativo cuesta significativamente más** que un Falso Positivo. Por ende, la prioridad táctica es incrementar el **Recall**.

---

## 4. Recomendaciones Estratégicas
1. **Ajuste del Umbral de Decisión:** Modificar el punto de corte probabilístico para incrementar el **Recall** antes del desembolso.
2. **Balanceo de Clases:** Implementar `class_weight='balanced'` o sobremuestreo (SMOTE) en futuros reentrenamientos.
3. **Automatización:** Este informe se actualiza dinámicamente en cada ejecución del pipeline en GitHub Actions.
"""
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(contenido_md)
    print(f"📝 Documento 'hallazgos.md' generado automáticamente en: '{ruta_salida}'")


def ejecutar_pipeline():
    print("==================================================")
    print("🚀 INICIANDO PIPELINE DE ML: RIESGO DE CRÉDITO")
    print("==================================================\n")
    
    inicio_tiempo = datetime.now()
    
    # 1. Ingesta
    df_raw = cargar_datos_huggingface(dataset_id="imodels/credit-card")
    
    # 2. Preprocesamiento (sin Data Leakage)
    X_train, X_test, y_train, y_test, scaler = preprocesar_datos(df_raw)
    
    # 3. Entrenamiento
    n_estimators = 100
    max_depth = 10
    random_state = 42
    modelo = entrenar_modelo_random_forest(
        X_train, y_train, 
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        random_state=random_state
    )
    
    # 4. Evaluación y Cálculo de Métricas
    metricas = evaluar_modelo(modelo, X_test, y_test, output_dir="informes")
    
    fin_tiempo = datetime.now()
    duracion = fin_tiempo - inicio_tiempo
    
    # 5. Generación Automática de Todos los Artefactos e Informes
    os.makedirs("informes", exist_ok=True)
    
    # A. Resumen en formato .txt
    ruta_resumen = os.path.join("informes", "resumen_ejecucion.txt")
    contenido_txt = f"""==================================================
INFORME DE EJECUCIÓN AUTOMÁTICA DEL PIPELINE DE ML
==================================================
Fecha y Hora de Ejecución: {fin_tiempo.strftime('%Y-%m-%d %H:%M:%S')}
Duración Total: {duracion.total_seconds():.2f} segundos

1. DATOS E INGESTA
------------------
Dataset Fuente: imodels/credit-card (Hugging Face)
Filas Ingestadas: {df_raw.shape[0]}
Columnas Totales: {df_raw.shape[1]}

2. PREPROCESAMIENTO Y DIVISIÓN
------------------------------
Muestras de Entrenamiento (Train): {X_train.shape[0]}
Muestras de Prueba (Test): {X_test.shape[0]}
Variables Predictoras (X): {X_train.shape[1]}
Escalado: StandardScaler (Fit en Train, Transform en Test - Sin Leakage)

3. MODELO Y HIPERPARÁMETROS
---------------------------
Algoritmo: Random Forest Classifier
n_estimators: {n_estimators}
max_depth: {max_depth}
random_state: {random_state}

4. MÉTRICAS DE EVALUACIÓN (TEST SET)
-----------------------------------
Exactitud (Accuracy):   {metricas['accuracy']:.4f}
Precisión (Precision):   {metricas['precision']:.4f}
Sensibilidad (Recall):   {metricas['recall']:.4f}
Puntuación F1 (F1-Score): {metricas['f1']:.4f}

Artefactos Generados Automáticamente:
- Gráfica Matriz de Confusión: informes/matriz_confusion.png
- Informe TXT de Ejecución: informes/resumen_ejecucion.txt
- Análisis de Hallazgos MD: informes/hallazgos.md

Status: PIPELINE EJECUTADO CON ÉXITO
==================================================
"""
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        f.write(contenido_txt)
        
    print(f"📄 Informe 'resumen_ejecucion.txt' generado en: '{ruta_resumen}'")
    
    # B. Informe de Hallazgos en formato .md
    generar_informe_hallazgos(metricas, ruta_salida="informes/hallazgos.md")
    
    print("\n==================================================")
    print("✅ PIPELINE COMPLETO Y ARTEFACTOS GENERADOS CON ÉXITO")
    print("==================================================")


if __name__ == "__main__":
    ejecutar_pipeline()