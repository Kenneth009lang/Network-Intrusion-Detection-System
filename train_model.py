#!/usr/bin/env python3
"""
train_model.py

GitHub-ready training script for the intrusion detection app (app.py).

Usage:
    python train_model.py
    (You'll be asked to paste the local CSV path)

Produces:
 - models/cnn_bilstm_model.h5
 - scaler.pkl
 - label_encoder.pkl
 - selected_features.txt
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# Config
# -------------------------
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
CNN_BILSTM_FILENAME = os.path.join(MODEL_DIR, "cnn_bilstm_model.h5")
SCALER_FILENAME = "scaler.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"
SELECTED_FEATURES_FILENAME = "selected_features.txt"
RANDOM_STATE = 42

# -------------------------
# Helper functions
# -------------------------
def try_read_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def drop_id_cols(df):
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

def build_cnn_bilstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# -------------------------
# Main
# -------------------------
def main():
    print("=== train_model.py ===")
    csv_path = input("Paste local CSV path (or press Enter to use 'data/cicids2017.csv'): ").strip()
    if not csv_path:
        csv_path = "data/cicids2017.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        sys.exit(1)

    print("Loading CSV...")
    df = try_read_csv(csv_path)
    print("Loaded shape:", df.shape)

    # basic cleaning
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = drop_id_cols(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropped {before - df.shape[0]} rows with NaN/Inf")

    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - df.shape[0]} duplicate rows")

    # find label column
    if 'Label' in df.columns:
        label_col = 'Label'
    elif 'label' in df.columns:
        label_col = 'label'
    else:
        label_col = input("Couldn't auto-detect label column. Enter the label/target column name: ").strip()
        if label_col not in df.columns:
            print(f"ERROR: provided label column '{label_col}' not in dataframe columns.")
            sys.exit(1)

    print("Using label column:", label_col)

    # select features (numeric columns excluding label)
    X_full = df.drop(columns=[label_col])
    # coerce to numeric and drop columns that become entirely NaN
    for c in X_full.columns:
        X_full[c] = pd.to_numeric(X_full[c], errors='coerce')
    X_full = X_full.dropna(axis=1, how='all')

    selected_features = X_full.columns.tolist()
    if not selected_features:
        print("ERROR: no numeric features detected after coercion.")
        sys.exit(1)

    # Save selected_features
    with open(SELECTED_FEATURES_FILENAME, "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")
    print(f"Wrote selected features ({len(selected_features)}) -> {SELECTED_FEATURES_FILENAME}")

    # prepare X and y
    X = X_full[selected_features].copy()
    y = df[label_col].copy()

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    with open(LABEL_ENCODER_FILENAME, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved label encoder -> {LABEL_ENCODER_FILENAME}. Classes: {list(le.classes_)}")

    # train/test split (before SMOTE)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE)
    print("Train/Test split:", X_train.shape, X_test.shape)

    # SMOTE balancing on training set
    print("Applying SMOTE to training data...")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE, train shape:", X_train_res.shape)

    # scaler fit on balanced training set
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)
    with open(SCALER_FILENAME, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler -> {SCALER_FILENAME}")

    # reshape for CNN/BiLSTM: (samples, features, 1)
    X_train_3d = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_3d = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # build and train cnn-bilstm
    num_classes = len(np.unique(y_enc))
    input_shape = (X_train_3d.shape[1], 1)
    model = build_cnn_bilstm(input_shape, num_classes)
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    print("Training CNN-BiLSTM (this may take a while)...")
    model.fit(X_train_3d, y_train_res, validation_data=(X_test_3d, y_test), epochs=20, batch_size=64, callbacks=[early_stop])

    # save model (exact name your app expects)
    model.save(CNN_BILSTM_FILENAME)
    print(f"Saved model -> {CNN_BILSTM_FILENAME}")

    # final evaluation
    eval_res = model.evaluate(X_test_3d, y_test, verbose=0)
    print(f"Final evaluation on test set: Loss={eval_res[0]:.4f}, Accuracy={eval_res[1]:.4f}")

    print("\nAll artifacts written. Your app.py should be able to load them now.")
    print("Artifacts produced:")
    print(" -", CNN_BILSTM_FILENAME)
    print(" -", SCALER_FILENAME)
    print(" -", LABEL_ENCODER_FILENAME)
    print(" -", SELECTED_FEATURES_FILENAME)

if __name__ == "__main__":
    # Lazy imports for a nicer error if deps are missing
    try:
        from sklearn.preprocessing import MinMaxScaler
    except Exception:
        print("Missing sklearn. Install required packages (see README).")
        raise
    main()
