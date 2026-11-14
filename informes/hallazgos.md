# 📊 Informe Profundizado de Hallazgos y Análisis del Modelo de Riesgo de Crédito

> **Nota:** Este informe fue generado **automáticamente por el pipeline de ML** el 2026-11-14 a las 17:39:34.

## 1. Pregunta de Negocio y Respuesta Ejecutiva

**Pregunta de negocio:**
> *¿Qué clientes titulares de tarjetas de crédito tienen una alta probabilidad de incumplir el pago del próximo mes, según su historial de pagos, saldos facturados, montos pagados y límite de crédito, para que el área de riesgos priorice la aprobación, la mitigación del cupo o la cobranza preventiva?*

**Respuesta ejecutiva (con ajuste de umbral aplicado):**
Mediante un modelo supervisado **Random Forest Classifier** (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`) con capacidad discriminante **ROC-AUC de 0.7744**, calibramos el punto de corte probabilístico desde el umbral estándar (`0.50`) hasta **`0.35`**. Con este ajuste, el sistema intercepta al **80.17% de los titulares que caerán en mora el próximo mes** (**853 de 1064 casos reales**), rescatando **218 morosos adicionales** frente al corte tradicional y reduciendo el costo financiero ponderado en un **0.69%**.

---

## 2. Rigor Metodológico y Auditoría de Datos

- **Ingesta automatizada:** Dataset `imodels/credit-card` desde Hugging Face con **24,000 registros y 34 variables**.
- **Saneamiento:** `0` valores nulos y **25 registros duplicados eliminados**, consolidando **23,975 clientes únicos**.
- **Cero *Data Leakage*:** División estratificada **80% Entrenamiento (19,180 muestras)** y **20% Prueba (4,795 muestras)** ejecutada antes del escalado (`StandardScaler`) y del ajuste del modelo.

---

## 3. Ajuste del Umbral de Decisión y Análisis del Error Financiero

En la banca, un **Falso Negativo (FN)** (aprobar o no alertar a un cliente que caerá en impago) conlleva pérdida directa de capital e intereses (costo relativo `5x`), mientras que un **Falso Positivo (FP)** (emitir una alerta preventiva sobre un cliente cumplidor) implica solo un costo operativo de revisión (costo relativo `1x`).

Para resolver esta asimetría, aplicamos la optimización sobre `predict_proba` evaluando múltiples puntos de corte y seleccionando aquel que maximiza el **F2-Score** (métrica que pondera el *Recall* el doble que la *Precision*):

| Umbral | Accuracy | Precision (Mora) | Recall (Mora) | Especificidad (Cumple) | F1-Score | F2-Score | Morosos Detectados (TP) | Morosos Omitidos (FN) | Costo Ponderado (5·FN + 1·FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.50** | 78.10% | 50.56% | **59.68%** | 83.36% | 54.74% | **57.60%** | 635 | 429 | 2,766 |
| **0.45** | 74.85% | 45.33% | **64.76%** | 77.73% | 53.33% | **59.64%** | 689 | 375 | 2,706 |
| **0.40** | 69.11% | 39.19% | **71.05%** | 68.56% | 50.52% | **61.12%** | 756 | 308 | 2,713 |
| **0.35** | 60.31% | 33.52% | **80.17%** | 54.65% | 47.27% | **62.71%** | 853 | 211 | 2,747 |

### Impacto Concreto del Ajuste (`0.50` → `0.35`)
- **Reducción de Falsos Negativos (Riesgo Crítico):** Los morosos no detectados bajan de **429** a **211** clientes (`+218` impagos interceptados a tiempo).
- **Incremento de Sensibilidad (Recall):** Sube de **59.68%** a **80.17%**, fortaleciendo la cobertura preventiva antes del vencimiento.
- **Costo de Oportunidad Controlado (FP):** La precisión pasa de **50.56%** a **33.52%**, manteniendo aprobados sin fricción a **2,039 clientes cumplidores** (`Especificidad: 54.65%`).

---

## 4. Variables Determinantes del Riesgo (Feature Importance)

Las 10 variables con mayor aporte predictivo dentro del bosque aleatorio son:

| Ranking | Variable | Importancia Relativa |
| :---: | :--- | :---: |
| 1 | `pay_0` | 22.43% |
| 2 | `pay_2` | 10.24% |
| 3 | `pay_3` | 6.79% |
| 4 | `pay_4` | 5.21% |
| 5 | `limit_bal` | 4.59% |
| 6 | `pay_amt1` | 4.56% |
| 7 | `pay_6` | 4.00% |
| 8 | `pay_amt3` | 3.76% |
| 9 | `bill_amt1` | 3.73% |
| 10 | `pay_amt2` | 3.59% |

> **Lectura financiera:** El comportamiento reciente de pagos (`PAY_0`, `PAY_2`) combinado con la relación entre cupo asignado (`LIMIT_BAL`), saldos facturados (`BILL_AMT1`) y abonos efectuados (`PAY_AMT1`) concentra la señal más fuerte de deterioro crediticio.

---

## 5. Matriz de Decisión Operativa para el Área de Riesgos

| Rango de Probabilidad (`p`) | Nivel de Riesgo | Acción Recomendada |
| :--- | :--- | :--- |
| `p < 0.35` | **Riesgo Bajo** | **Priorizar aprobación:** originación ágil sin fricción comercial y elegibilidad para campañas de fidelización. |
| `0.35 <= p < 0.65` | **Riesgo Moderado** | **Mitigación de cupo:** congelar ampliaciones automáticas de línea de crédito y monitorear utilización de saldo. |
| `p >= 0.65` | **Riesgo Alto** | **Cobranza preventiva:** gestión proactiva multicanal antes de la fecha de corte y acuerdos tempranos de pago. |

---

## 6. Limitaciones Técnicas y Próximos Pasos
1. **Carga operativa por falsas alarmas:** Al reducir el umbral a `0.35` se generan `1692` alertas preventivas sobre buenos pagadores; se recomienda automatizar las alertas de riesgo moderado vía SMS/correo y concentrar gestión humana en probabilidades mayores a `0.65`.
2. **Variables externas:** El dataset no incluye variables macroeconómicas (inflación, desempleo) ni relación deuda/ingreso actualizada; integrarlas elevaría aún más la precisión.
3. **Automatización Continua:** Tanto el umbral óptimo (`0.35`) como este reporte se recalculan y empaquetan automáticamente en cada ejecución de GitHub Actions.
