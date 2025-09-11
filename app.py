import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os
import shap
import matplotlib.pyplot as plt
from lime import lime_tabular

# Set page configuration for a wider layout
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# --- Function to load the saved objects with caching ---
@st.cache_resource
def load_resources():
    """
    Loads the trained model, scaler, label encoder, and selected features list.
    """
    try:
        # Load the Keras model
        model = tf.keras.models.load_model('cnn_bilstm_model.h5')
        
        # Load the scaler
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Load the label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
            
        # Load the selected features list
        with open('selected_features.txt', 'r') as f:
            selected_features = [line.strip() for line in f.readlines()]
            
        return model, scaler, label_encoder, selected_features

    except FileNotFoundError as e:
        st.error(f"Error: Missing a required model file: {e}. "
                 "Please ensure all files are in the same directory.")
        st.stop()
        
# Load all the necessary resources when the app starts
model, scaler, label_encoder, selected_features = load_resources()

# --- Streamlit App User Interface (UI) ---
st.title("🛡️ Network Intrusion Detection System")
st.markdown("This application classifies network traffic using a deep learning model.")
st.markdown("---")

st.header("Upload Network Traffic Data (CSV)")
st.info("Please ensure your CSV file contains the following features (columns) for accurate prediction: " + ", ".join(selected_features))
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# --- Prediction Logic ---
if uploaded_file is not None:
    try:
        df_new = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df_new.head())
        
        missing_features = [f for f in selected_features if f not in df_new.columns]
        if missing_features:
            st.error(f"Error: The uploaded file is missing the following required columns: {', '.join(missing_features)}")
            st.stop()
        
        st.subheader("Making Predictions...")
        with st.spinner('Preprocessing data and making predictions...'):
            df_new_selected = df_new[selected_features]
            df_new_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_new_selected.dropna(inplace=True)
            
            X_new_scaled = scaler.transform(df_new_selected)
            X_new_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
            
            predictions_probs = model.predict(X_new_reshaped, verbose=0)
            predictions = np.argmax(predictions_probs, axis=1)
            predicted_labels = label_encoder.inverse_transform(predictions)

        st.success("✅ Predictions complete!")
        
        # --- Display Results ---
        st.header("Prediction Results")
        results_df = df_new.loc[df_new_selected.index].copy()
        results_df['Predicted_Label'] = predicted_labels
        
        st.subheader("Top 5 Predictions")
        st.dataframe(results_df.head(5))
        
        st.subheader("Prediction Distribution")
        prediction_counts = results_df['Predicted_Label'].value_counts().sort_index()
        st.bar_chart(prediction_counts)
        
        st.markdown("---")
        st.write("Overall Prediction Summary:")
        st.dataframe(prediction_counts.to_frame(name='Count'))

        # --- XAI Section ---
        st.header("🔎 Model Explainability (XAI)")

        # --- SHAP ---
        if st.button("Generate SHAP Explanations"):
            st.subheader("SHAP Explanations")
            explainer = shap.Explainer(model, X_new_reshaped[:200])  # limit for speed
            shap_values = explainer(X_new_reshaped[:200])

            st.write("**Global Feature Importance**")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_new_scaled[:200], feature_names=selected_features, show=False)
            st.pyplot(fig)

            st.write("**Local Explanation for First Row**")
            fig2, ax2 = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig2)

        # --- LIME ---
        if st.button("Generate LIME Explanation (Row 1)"):
            st.subheader("LIME Explanation")
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=X_new_scaled,
                feature_names=selected_features,
                class_names=label_encoder.classes_,
                mode="classification"
            )

            i = 0  # First row for demo
            exp = explainer.explain_instance(
                X_new_scaled[i],
                lambda x: model.predict(x.reshape(-1, X_new_scaled.shape[1], 1)),
                num_features=10
            )

            st.write(exp.as_list())
            fig3 = exp.as_pyplot_figure()
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}. Please check your file and try again.")
