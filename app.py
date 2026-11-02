"""Simulador local de predicción de riesgo de mora."""

from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


ROOT = Path(__file__).parent
REPORTS = ROOT / "informes"
MODEL_PATH = REPORTS / "modelo_riesgo.joblib"
SUMMARY_PATH = REPORTS / "resumen_ejecucion.txt"
CONFUSION_PATH = REPORTS / "matriz_confusion.png"

st.set_page_config(page_title="Simulador de riesgo", page_icon="💳", layout="wide")
st_autorefresh(interval=2000, key="pipeline_file_watcher")


def _signature() -> tuple[int, int]:
    files = [MODEL_PATH, SUMMARY_PATH, CONFUSION_PATH]
    return tuple(path.stat().st_mtime_ns if path.exists() else 0 for path in files)


@st.cache_resource
def load_bundle(signature: tuple[int, int]):
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def read_metrics() -> dict[str, float]:
    if not SUMMARY_PATH.exists():
        return {}
    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    labels = {
        "Accuracy": "Exactitud (Accuracy)",
        "Precision": "Precisión (Precision)",
        "Recall": "Sensibilidad (Recall)",
        "F1": "Puntuación F1 (F1-Score)",
    }
    metrics = {}
    for name, label in labels.items():
        match = re.search(rf"{re.escape(label)}:\s+([0-9.]+)", summary)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


st.title("💳 Simulador de riesgo de crédito")
st.caption("La aplicación revisa los artefactos del pipeline cada 2 segundos.")

signature = _signature()
bundle = load_bundle(signature)

if bundle is None:
    st.warning(
        "Todavía no existe el modelo generado. Ejecuta `python main.py` y vuelve a esperar la actualización."
    )
    st.stop()

metrics = read_metrics()
if metrics:
    history = st.session_state.setdefault("metrics_history", [])
    history_signature = tuple(metrics.items())
    if not history or history[-1]["signature"] != history_signature:
        history.append(
            {
                "signature": history_signature,
                "ejecucion": len(history) + 1,
                **metrics,
            }
        )

model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
defaults = bundle["defaults"]

st.info(
    "Los cambios en `main.py`, el modelo o los informes se reflejan después de ejecutar "
    "`python main.py`; la aplicación detecta los nuevos artefactos automáticamente."
)

with st.form("risk_form"):
    st.subheader("Datos del cliente")
    values = {}
    columns = st.columns(3)
    for index, feature in enumerate(features):
        default = float(defaults.get(feature, 0.0))
        values[feature] = columns[index % 3].number_input(
            feature, value=default, format="%.2f", key=f"input_{feature}"
        )
    submitted = st.form_submit_button("Evaluar riesgo")

if submitted:
    row = pd.DataFrame([values], columns=features)
    scaled_row = pd.DataFrame(scaler.transform(row), columns=features)
    prediction = int(model.predict(scaled_row)[0])
    probability = float(model.predict_proba(scaled_row)[0, 1])

    if prediction == 1:
        st.error(f"Riesgo alto de mora. Probabilidad estimada: {probability:.2%}")
    else:
        st.success(f"Riesgo bajo de mora. Probabilidad estimada: {probability:.2%}")

if metrics:
    st.subheader("Métricas de la última ejecución")
    metric_columns = st.columns(4)
    for column, (name, value) in zip(metric_columns, metrics.items()):
        column.metric(name, f"{value:.2%}")

    chart_data = pd.DataFrame([metrics]).T.rename(columns={0: "valor"})
    st.bar_chart(chart_data, y="valor")

    if len(st.session_state.get("metrics_history", [])) > 1:
        st.subheader("Cambios entre ejecuciones")
        history_data = pd.DataFrame(st.session_state["metrics_history"]).drop(
            columns="signature"
        )
        st.line_chart(history_data.set_index("ejecucion"))

if CONFUSION_PATH.exists():
    st.subheader("Matriz de confusión")
    st.image(
        str(CONFUSION_PATH),
        caption="Aciertos y errores del modelo en el conjunto de prueba",
        width="stretch",
    )

st.subheader("Último resumen generado")
if SUMMARY_PATH.exists():
    st.code(SUMMARY_PATH.read_text(encoding="utf-8"), language="text")
