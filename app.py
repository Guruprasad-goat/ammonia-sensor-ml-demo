import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Sensor ML Demo", page_icon="📡", layout="centered")
st.title("📡 Simulated Sensor Pattern Classifier")
st.caption("Educational prototype — trained on generated synthetic data")

st.error(
    "NOT A REAL AMMONIA DETECTOR. The training data is synthetic and arbitrary. "
    "Do not connect this result to an alarm, make safety decisions from it, "
    "or test it using ammonia or other chemicals."
)

@st.cache_resource
def load_assets():
    model = joblib.load("sensor_demo_model.joblib")
    with open("model_metadata.json", "r") as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_assets()
except Exception as e:
    st.error("Model files are missing. Upload sensor_demo_model.joblib and model_metadata.json to this repository.")
    st.stop()

st.subheader("Enter four arbitrary normalized demo values")
st.write("These sliders are for demonstrating the software workflow only; they are not sensor calibration values.")

cols = st.columns(2)
values = []
for i, feature in enumerate(meta["features"]):
    with cols[i % 2]:
        values.append(st.slider(feature.replace("_", " ").title(), -4.0, 4.0, 0.0, 0.1))

if st.button("Run demo prediction", type="primary", use_container_width=True):
    row = pd.DataFrame([values], columns=meta["features"])
    prediction = int(model.predict(row)[0])
    label = meta["labels"][str(prediction)]
    probabilities = model.predict_proba(row)[0]

    st.subheader("Model output")
    if prediction == 0:
        st.success(label)
    elif prediction == 1:
        st.warning(label)
    else:
        st.error(label)

    st.write("Illustrative model score distribution (not a calibrated safety probability):")
    for class_id, score in enumerate(probabilities):
        st.write(f"{meta['labels'][str(class_id)]}: {score:.1%}")
        st.progress(float(score))

st.divider()
st.caption("Model: Random Forest • Dataset: 6,000 generated synthetic rows • Train/test split: 80/20")
st.caption("A high test score here only shows that the model learned patterns in its own synthetic dataset.")
