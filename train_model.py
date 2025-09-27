"""
train_model.py
-------------------------------------
Train CNN and CNN-BiLSTM models on the CICIDS2017 dataset.

✅ Reads CSV from local 'data/' folder
✅ Preprocesses data (encoding, scaling, feature selection)
✅ Trains both CNN and CNN-BiLSTM
✅ Saves model artifacts (optional if you already have them)
✅ GitHub-ready: documented, readable, reproducible

Author: Your Name
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# 1. Load Dataset
# --------------------------
DATA_PATH = "data/cicids2017.csv"
assert os.path.exists(DATA_PATH), f"❌ Dataset not found at {DATA_PATH}"

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded shape: {df.shape}")

# --------------------------
# 2. Basic Cleaning
# --------------------------
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Assuming last column is the label
label_col = df.columns[-1]
X = df.drop(columns=[label_col])
y = df[label_col]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder (optional)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# --------------------------
# 3. Feature Scaling
# --------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler (optional)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature names
with open("selected_features.txt", "w") as f:
    for col in X.columns:
        f.write(col + "\n")

# --------------------------
# 4. Handle Class Imbalance
# --------------------------
print("🔁 Balancing data with SMOTE...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_encoded)
print("✅ After SMOTE:", X_res.shape, y_res.shape)

# --------------------------
# 5. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Reshape for CNN/LSTM input
X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --------------------------
# 6. Define Models
# --------------------------

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------
# 7. Train Models
# --------------------------
num_classes = len(np.unique(y_res))
input_shape = (X_train_3d.shape[1], 1)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\n🚀 Training CNN model...")
cnn = build_cnn(input_shape, num_classes)
cnn.fit(X_train_3d, y_train, validation_data=(X_test_3d, y_test),
        epochs=10, batch_size=256, callbacks=[early_stop], verbose=1)
cnn.save("models/cnn_model.h5")
print("✅ CNN model saved.")

print("\n🚀 Training CNN-BiLSTM model...")
cnn_bilstm = build_cnn_bilstm(input_shape, num_classes)
cnn_bilstm.fit(X_train_3d, y_train, validation_data=(X_test_3d, y_test),
               epochs=10, batch_size=256, callbacks=[early_stop], verbose=1)
cnn_bilstm.save("models/cnn_bilstm_model.h5")
print("✅ CNN-BiLSTM model saved.")

# --------------------------
# 8. Evaluate
# --------------------------
cnn_eval = cnn.evaluate(X_test_3d, y_test, verbose=0)
bilstm_eval = cnn_bilstm.evaluate(X_test_3d, y_test, verbose=0)

print("\n📊 Model Evaluation:")
print(f"CNN Accuracy: {cnn_eval[1]:.4f}")
print(f"CNN-BiLSTM Accuracy: {bilstm_eval[1]:.4f}")

print("\n🎉 Training complete! Artifacts ready for app.py")
