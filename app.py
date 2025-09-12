import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import datetime
import zipfile
from io import BytesIO

st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# ----------------------------
# Cache Clearing Function
# ----------------------------
def clear_cache():
    st.cache_resource.clear()
    st.cache_data.clear()

with st.sidebar:
    st.button("Clear Cache", on_click=clear_cache)
    st.markdown("---")
    st.write("If you encounter errors after uploading a new file, try clearing the cache.")

# ----------------------------
# Load resources
# ----------------------------
@st.cache_resource(show_spinner="Loading model files...")
def load_resources():
    try:
        model = tf.keras.models.load_model('cnn_bilstm_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('selected_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
        return model, scaler, label_encoder, selected_features
    except FileNotFoundError as e:
        st.error(f"Missing required file: {e}")
        st.stop()

model, scaler, label_encoder, selected_features = load_resources()

# ----------------------------
# Helper Functions
# ----------------------------
def predict_2d_to_3d(x_2d):
    x_arr = np.array(x_2d, dtype=np.float32)
    x_3d = x_arr.reshape(x_arr.shape[0], x_arr.shape[1], 1)
    return model.predict(x_3d, verbose=0)

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# ----------------------------
# SHAP Explainer for Multiclass
# ----------------------------
@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_shap_explainer_multiclass(X_scaled, nsamples_bg=100):
    rng = np.random.default_rng(seed=42)
    n_samples_bg = min(nsamples_bg, X_scaled.shape[0])
    bg_idx = rng.choice(X_scaled.shape[0], size=n_samples_bg, replace=False)
    background = X_scaled[bg_idx]

    explainer = shap.KernelExplainer(predict_2d_to_3d, background)
    return explainer

# ----------------------------
# LIME Explainer
# ----------------------------
@st.cache_resource(show_spinner="Preparing LIME explainer...")
def get_lime_explainer(data, feature_names, class_names):
    return LimeTabularExplainer(data, feature_names=feature_names, class_names=class_names, mode='classification')

# ----------------------------
# SHAP & LIME Plotting
# ----------------------------
def plot_shap_local_multiclass(shap_explainer, X_scaled, row_idx, preds, feature_names):
    shap_vals = shap_explainer.shap_values(X_scaled[row_idx:row_idx+1], nsamples=100)
    pred_class_idx = int(preds[row_idx])
    shap_for_row = np.array(shap_vals[pred_class_idx])[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    feat = np.array(feature_names)
    order = np.argsort(np.abs(shap_for_row))[::-1]
    ax.barh(feat[order][:20][::-1], shap_for_row[order][:20][::-1])
    ax.set_xlabel("SHAP value (contribution)")
    ax.set_title(f"Local SHAP for Row {row_idx} (Class: {pred_class_idx})")
    plt.tight_layout()
    return fig

def plot_lime_local(lime_exp, title="LIME Local Explanation"):
    fig = lime_exp.as_pyplot_figure()
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def plot_shap_global(shap_explainer, data, feature_names, n_samples=200):
    subset_n = min(n_samples, data.shape[0])
    rng = np.random.default_rng(seed=42)
    global_idx = rng.choice(data.shape[0], size=subset_n, replace=False)
    shap_vals_global = shap_explainer.shap_values(data[global_idx], nsamples=100)

    mean_abs_per_class = np.array([np.mean(np.abs(sv), axis=0) for sv in shap_vals_global])
    mean_abs_across_classes = np.mean(mean_abs_per_class, axis=0)

    feat = np.array(feature_names)
    orderg = np.argsort(mean_abs_across_classes)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feat[orderg][:30][::-1], mean_abs_across_classes[orderg][:30][::-1])
    ax.set_title("Global Feature Importance (Mean |SHAP| across classes)")
    ax.set_xlabel("Mean |SHAP value|")
    plt.tight_layout()
    return fig

# ----------------------------
# Main App UI
# ----------------------------
st.title("🛡️ Network Intrusion Detection System — SHAP & LIME")
st.markdown("Upload a CSV file with network traffic data to get predictions and explanations.")
st.markdown("---")
st.info("Required features: " + ", ".join(selected_features))

uploaded_file = st.file_uploader("Upload network traffic CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# ----------------------------
# Read, preprocess, predict
# ----------------------------
with st.spinner("Processing file and generating predictions..."):
    df = pd.read_csv(uploaded_file)
    missing = [c for c in selected_features if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing)}")
        st.stop()

    df_sel = df[selected_features].copy()
    df_sel.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sel.dropna(inplace=True)
    if df_sel.empty:
        st.error("No valid data found after handling missing values.")
        st.stop()

    X_scaled = scaler.transform(df_sel)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    preds_probs = model.predict(X_reshaped, verbose=0)
    preds = np.argmax(preds_probs, axis=1)
    preds_labels = label_encoder.inverse_transform(preds)

    results_df = df.loc[df_sel.index].copy()
    results_df['Predicted_Label'] = preds_labels
    results_df['Predicted_Probability'] = np.max(preds_probs, axis=1)

# ----------------------------
# Initialize explainers
# ----------------------------
shap_explainer = get_shap_explainer_multiclass(X_scaled, nsamples_bg=100)
lime_explainer = get_lime_explainer(X_scaled, selected_features, label_encoder.classes_)

# ----------------------------
# Tabs: Predictions & Explainability
# ----------------------------
tab1, tab2 = st.tabs(["📊 Predictions", "🔎 Explainability & Export"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(results_df.head())
    st.subheader("Prediction Distribution")
    counts = results_df['Predicted_Label'].value_counts()
    st.bar_chart(counts)
    st.dataframe(counts.to_frame(name='Count'))

with tab2:
    st.header("Model Explainability (SHAP + LIME)")
    unique_labels = results_df['Predicted_Label'].unique().tolist()
    selected_class = st.selectbox("Select predicted class to inspect:", unique_labels)
    class_rows = results_df[results_df['Predicted_Label'] == selected_class]

    if class_rows.empty:
        st.warning("No rows for the selected class.")
    else:
        selected_row_index = st.selectbox("Select a row index:", class_rows.index.tolist())
        numpy_row_idx = np.where(df_sel.index.values == selected_row_index)[0][0]

        st.subheader("Selected Row Details")
        st.dataframe(results_df.loc[[selected_row_index]])

        col_shap, col_lime = st.columns(2)

        with col_shap:
            st.subheader("SHAP (KernelExplainer)")
            fig_shap = plot_shap_local_multiclass(shap_explainer, X_scaled, numpy_row_idx, preds, selected_features)
            st.pyplot(fig_shap)
            plt.close(fig_shap)

        with col_lime
