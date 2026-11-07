import os
from datetime import datetime
import joblib
from src.ingesta import cargar_datos_huggingface
from src.preprocesamiento import preprocesar_datos
from src.modelo import comparar_configuraciones, entrenar_modelo_random_forest
from src.evaluacion import evaluar_modelo


def generar_informe_hallazgos(metricas: dict, ruta_salida: str = "informes/hallazgos.md"):
    """
    Genera automáticamente el informe de hallazgos en formato Markdown
    inyectando las métricas calculadas en tiempo real.
    """
    contenido_md = f"""# 📊 Informe de Hallazgos y Análisis del Modelo de Riesgo de Crédito

> **Nota:** Este informe fue generado **automáticamente por el pipeline de ML** el {datetime.now().strftime('%Y-%m-%d a las %H:%M:%S')}.

## 1. Resumen Ejecutivo
**Pregunta de negocio:** ¿Qué clientes titulares de tarjetas de crédito tienen una alta probabilidad de incumplir el pago del próximo mes para priorizar aprobación, mitigación de cupo o cobranza preventiva?

Este documento analiza el desempeño del modelo **Random Forest Classifier** entrenado sobre el dataset `imodels/credit-card`. El modelo identifica señales útiles de riesgo, pero no debe automatizar por sí solo el rechazo de clientes porque todavía deja algunos casos de mora sin detectar.

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

En la prueba, el modelo obtuvo **{metricas['confusion_matrix'][1, 1]} verdaderos positivos** y dejó **{metricas['confusion_matrix'][1, 0]} falsos negativos**: detectó una parte de los clientes que cayeron en mora, pero omitió otra parte relevante.

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
2. **Balanceo de Clases:** El entrenamiento actual utiliza `class_weight='balanced'` para dar mayor peso a la clase minoritaria de clientes morosos.
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
    
    # 3. Comparación de configuraciones en una validación interna del train
    configuraciones = [
        {"n_estimators": 50, "max_depth": 8, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 10, "class_weight": "balanced"},
        {"n_estimators": 100, "max_depth": 15, "class_weight": "balanced"},
    ]
    comparacion = comparar_configuraciones(
        X_train, y_train, configuraciones=configuraciones, random_state=42
    )
    configuracion_elegida = max(
        comparacion,
        key=lambda resultado: (resultado["f1"], resultado["recall"]),
    )

    # 4. Entrenamiento final sobre todo el conjunto de entrenamiento
    n_estimators = configuracion_elegida["n_estimators"]
    max_depth = configuracion_elegida["max_depth"]
    class_weight = configuracion_elegida["class_weight"]
    random_state = 42
    modelo = entrenar_modelo_random_forest(
        X_train, y_train, 
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        class_weight=class_weight,
        random_state=random_state
    )

    # Guardar modelo y transformador para el simulador interactivo local.
    df_para_defaults = df_raw.drop_duplicates().dropna()
    columnas_modelo = list(X_train.columns)
    target_col = "default.payment.next.month"
    defaults = (
        df_para_defaults.drop(columns=[target_col], errors="ignore")
        .reindex(columns=columnas_modelo)
        .median()
        .to_dict()
    )
    os.makedirs("informes", exist_ok=True)
    joblib.dump(
        {"model": modelo, "scaler": scaler, "features": columnas_modelo, "defaults": defaults},
        "informes/modelo_riesgo.joblib",
    )
    
    # 5. Evaluación y Cálculo de Métricas
    metricas = evaluar_modelo(modelo, X_test, y_test, output_dir="informes")
    
    fin_tiempo = datetime.now()
    duracion = fin_tiempo - inicio_tiempo
    
    # 6. Generación Automática de Todos los Artefactos e Informes
    os.makedirs("informes", exist_ok=True)
    df_limpio = df_raw.drop_duplicates().dropna()
    columnas = ", ".join(
        f"{col} ({df_raw[col].dtype})" for col in df_raw.columns
    )
    muestra = df_raw.head(3).to_string(index=False)
    muestra_escalada = X_train.head(3).round(3).to_string(index=False)
    distribucion = y_test.value_counts(normalize=True).sort_index()
    comparacion_txt = "\n".join(
        f"- n_estimators={r['n_estimators']}, max_depth={r['max_depth']}: "
        f"recall={r['recall']:.4f}, F1={r['f1']:.4f}"
        for r in comparacion
    )
    
    # A. Resumen en formato .txt
    ruta_resumen = os.path.join("informes", "resumen_ejecucion.txt")
    contenido_txt = f"""==================================================
INFORME DE EJECUCIÓN AUTOMÁTICA DEL PIPELINE DE ML
==================================================
Fecha y Hora de Ejecución: {fin_tiempo.strftime('%Y-%m-%d %H:%M:%S')}
Duración Total: {duracion.total_seconds():.2f} segundos

PREGUNTA DE NEGOCIO
-------------------
¿Qué clientes tienen alta probabilidad de incurrir en mora el próximo mes
para priorizar aprobación, mitigación de cupo o cobranza preventiva?

1. DATOS E INGESTA
------------------
Dataset Fuente: imodels/credit-card (Hugging Face)
Filas Ingestadas: {df_raw.shape[0]}
Columnas Totales: {df_raw.shape[1]}
Split/configuración: train
Columnas y tipos:
{columnas}
Muestra de 3 registros:
{muestra}

2. PREPROCESAMIENTO Y DIVISIÓN
------------------------------
Nulos iniciales: {df_raw.isnull().sum().sum()}
Duplicados iniciales: {df_raw.duplicated().sum()}
Filas después de limpieza: {df_limpio.shape[0]}
Muestras de Entrenamiento (Train): {X_train.shape[0]}
Muestras de Prueba (Test): {X_test.shape[0]}
Variables Predictoras (X): {X_train.shape[1]}
Distribución de clases en test: clase 0={distribucion.get(0, 0):.2%}, clase 1={distribucion.get(1, 0):.2%}
Escalado: StandardScaler (fit en Train, transform en Test - Sin Leakage).
Aunque Random Forest no requiere escalado, se conserva para dejar las variables
en una escala comparable y mantener un pipeline reproducible.
Muestra de predictoras después del escalado (primeras 3 filas de train):
{muestra_escalada}

3. MODELO Y HIPERPARÁMETROS
---------------------------
Algoritmo: Random Forest Classifier
n_estimators: {n_estimators}
max_depth: {max_depth}
class_weight: {class_weight}
random_state: {random_state}
Comparación en validación interna del train:
{comparacion_txt}
Configuración elegida: {configuracion_elegida}

4. MÉTRICAS DE EVALUACIÓN (TEST SET)
-----------------------------------
Exactitud (Accuracy):   {metricas['accuracy']:.4f}
Precisión (Precision):   {metricas['precision']:.4f}
Sensibilidad (Recall):   {metricas['recall']:.4f}
Puntuación F1 (F1-Score): {metricas['f1']:.4f}

Artefactos Generados Automáticamente:
- Gráfica Matriz de Confusión: informes/matriz_confusion.png
- Modelo para simulador: informes/modelo_riesgo.joblib
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