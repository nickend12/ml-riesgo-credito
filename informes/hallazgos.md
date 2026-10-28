# 📊 Informe de Hallazgos y Análisis del Modelo de Riesgo de Crédito

> **Nota:** Este informe fue generado **automáticamente por el pipeline de ML** el 2026-10-28 a las 02:57:55.

## 1. Resumen Ejecutivo
Este documento analiza el desempeño del modelo **Random Forest Classifier** entrenado sobre el dataset `imodels/credit-card`. El objetivo es optimizar la toma de decisiones financieras en la aprobación de crédito, equilibrando la mitigación del riesgo con el crecimiento de la cartera.

---

## 2. Métricas de Evaluación Obtenidas (Conjunto de Prueba)

| Métrica | Valor Calculado | Interpretación de Negocio |
| :--- | :--- | :--- |
| **Accuracy (Exactitud)** | **81.84%** | Clasificación correcta general (cumplidos y morosos). |
| **Precision (Precisión)** | **67.58%** | De las alarmas de mora emitidas, este porcentaje realmente cayó en mora. |
| **Recall (Sensibilidad)** | **34.87%** | Porcentaje total de clientes en mora capturados por el modelo. |
| **F1-Score** | **46.00%** | Balance armónico entre Precisión y Sensibilidad. |

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
