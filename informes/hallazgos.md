# 📊 Informe de Hallazgos y Análisis del Modelo de Riesgo de Crédito

> **Nota:** Este informe fue generado **automáticamente por el pipeline de ML** el 2026-11-07 a las 23:15:15.

## 1. Resumen Ejecutivo
**Pregunta de negocio:** ¿Qué clientes titulares de tarjetas de crédito tienen una alta probabilidad de incumplir el pago del próximo mes para priorizar aprobación, mitigación de cupo o cobranza preventiva?

Este documento analiza el desempeño del modelo **Random Forest Classifier** entrenado sobre el dataset `imodels/credit-card`. El modelo identifica señales útiles de riesgo, pero no debe automatizar por sí solo el rechazo de clientes porque todavía deja algunos casos de mora sin detectar.

---

## 2. Métricas de Evaluación Obtenidas (Conjunto de Prueba)

| Métrica | Valor Calculado | Interpretación de Negocio |
| :--- | :--- | :--- |
| **Accuracy (Exactitud)** | **78.10%** | Clasificación correcta general (cumplidos y morosos). |
| **Precision (Precisión)** | **50.56%** | De las alarmas de mora emitidas, este porcentaje realmente cayó en mora. |
| **Recall (Sensibilidad)** | **59.68%** | Porcentaje total de clientes en mora capturados por el modelo. |
| **F1-Score** | **54.74%** | Balance armónico entre Precisión y Sensibilidad. |

---

## 3. Matriz de Confusión y Análisis del Error Financiero

En la prueba, el modelo obtuvo **635 verdaderos positivos** y dejó **429 falsos negativos**: detectó una parte de los clientes que cayeron en mora, pero omitió otra parte relevante.

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
