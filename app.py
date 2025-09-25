import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------------------
# 1. Load model & preprocessing
# -------------------------------
@st.cache_resource
def load_cnn_bilstm_model():
    return load_model("model.h5")

@st.cache_resource
def load_scaler():
    try:
        return joblib.load("scaler.pkl")
    except:
        return None

model = load_cnn_bilstm_model()
scaler = load_scaler()

# -------------------------------
# 2. Chunked CSV reader
# -------------------------------
@st.cache_data
def load_csv_in_chunks(file_path, chunk_size=50000, max_rows=100000):
    chunks = []
    row_count = 0
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        row_count += len(chunk)
        if row_count >= max_rows:  # limit preview size
            break
    df = pd.concat(chunks, ignore_index=True)
    return df

# -------------------------------
# 3. App UI
# -------------------------------
st.title("🔐 CNN-BiLSTM Intrusion Detection with Explainability")

st.sidebar.header("⚙️ Settings")
csv_file = st.sidebar.text_input("CSV file path", "CICIDS2017_filtered.csv")
max_rows = st.sidebar.number_input("Rows to load (limit)", min_value=10000, max_value=200000, value=50000, step=10000)

if csv_file:
    st.write(f"Loading up to {max_rows} rows from **{csv_file}**...")
    df = load_csv_in_chunks(csv_file, chunk_size=20000, max_rows=max_rows)
    st.write("✅ Data loaded:", df.shape)
    st.dataframe(df.head(20))

    # -------------------------------
    # 4. User selects a row
    # -------------------------------
    idx = st.number_input("Pick a row index for prediction & explanation", min_value=0, max_value=len(df)-1, value=0)
    input_row = df.drop(columns=["Label"], errors="ignore").iloc[[idx]]

    # -------------------------------
    # 5. Preprocess
    # -------------------------------
    if scaler:
        input_scaled = scaler.transform(input_row)
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        input_scaled = scaler.fit_transform(input_row)

    # CNN-BiLSTM expects 3D input
    X_input = input_scaled.reshape((input_scaled.shape[0], input_scaled.shape[1], 1))

    # -------------------------------
    # 6. Prediction
    # -------------------------------
    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.subheader("🔎 Prediction Result")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.bar_chart(prediction[0])

    # -------------------------------
    # 7. SHAP Explainability
    # -------------------------------
    st.subheader("📊 SHAP Explainability")

    # KernelExplainer for tabular models
    explainer = shap.KernelExplainer(model.predict, input_scaled[:50])  # background set
    shap_values = explainer.shap_values(input_scaled)

    st.write("Top features contributing to prediction:")
    shap.summary_plot(shap_values, input_row, plot_type="bar", show=False)
    st.pyplot(bbox_inches="tight")

    # -------------------------------
    # 8. LIME Explainability
    # -------------------------------
    st.subheader("🟢 LIME Explanation")

    feature_names = input_row.columns.tolist()
    class_names = [str(i) for i in range(prediction.shape[1])]

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(df.drop(columns=["Label"], errors="ignore")),
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    exp = lime_explainer.explain_instance(
        data_row=input_scaled[0],
        predict_fn=model.predict
    )

    st.write("LIME feature contributions:")
    st.write(exp.as_list())
    st.pyplot(exp.as_pyplot_figure())
