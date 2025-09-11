import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Set page configuration for a wider layout
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# --- Function to load the saved objects with caching ---
# The @st.cache_resource decorator ensures that the model and other large
# objects are loaded only once, improving app performance.
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
        # Read the uploaded file into a pandas DataFrame
        df_new = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(df_new.head())
        
        # Check if all required features are in the uploaded file
        missing_features = [f for f in selected_features if f not in df_new.columns]
        if missing_features:
            st.error(f"Error: The uploaded file is missing the following required columns: {', '.join(missing_features)}")
            st.stop()
        
        st.subheader("Making Predictions...")
        with st.spinner('Preprocessing data and making predictions...'):
            # 1. Select the features used during training
            df_new_selected = df_new[selected_features]
            
            # 2. Handle missing values and infinities
            df_new_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_new_selected.dropna(inplace=True)
            
            # 3. Scale the features using the loaded scaler
            X_new_scaled = scaler.transform(df_new_selected)
            
            # 4. Reshape data for the CNN-BiLSTM model
            # The model expects a 3D array: (samples, timesteps, features)
            X_new_reshaped = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
            
            # 5. Make predictions
            predictions_probs = model.predict(X_new_reshaped, verbose=0)
            predictions = np.argmax(predictions_probs, axis=1)
            predicted_labels = label_encoder.inverse_transform(predictions)

        st.success("✅ Predictions complete!")
        
        # --- Display Results ---
        st.header("Prediction Results")
        
        # Add the predictions to a DataFrame for display
        results_df = df_new.loc[df_new_selected.index].copy()
        results_df['Predicted_Label'] = predicted_labels
        
        st.subheader("Top 5 Predictions")
        st.dataframe(results_df.head(5))
        
        # Show prediction distribution
        st.subheader("Prediction Distribution")
        prediction_counts = results_df['Predicted_Label'].value_counts().sort_index()
        st.bar_chart(prediction_counts)
        
        st.markdown("---")
        st.write("Overall Prediction Summary:")
        st.dataframe(prediction_counts.to_frame(name='Count'))

    except KeyError as e:
        st.error(f"Error: The uploaded file is missing a required column: {e}.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}. Please check your file and try again.")
