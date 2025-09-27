from IPython import get_ipython
from IPython.display import display

# install stuff we need (colab usually has some of these)
!pip install tensorflow
!pip install imbalanced-learn scikit-plot shap lime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, time

# basics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# encoders & scalers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# evaluation helpers
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# quick baseline model
from sklearn.ensemble import RandomForestClassifier

# handle class imbalance
from imblearn.over_sampling import SMOTE
# imbalance handling
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import tensorflow as tf
# keras model setup
from tensorflow.keras.models import Sequential

# layers i’ll actually use
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional,
    Conv1D,
    MaxPooling1D,
    Dropout   # regularization trick
)

# training helpers
from tensorflow.keras.callbacks import EarlyStopping
# explainability
import shap, lime
import lime.lime_tabular

# seeds (for reproducibility, at least somewhat)
np.random.seed(42)
tf.random.set_seed(42)

print("all libs ready to go")

# --- load CICIDS2017 data ---
from google.colab import drive
drive.mount('/content/drive')

data_path = '/content/drive/My Drive/ml cve/MachineLearningCVE'
print("reading from:", data_path)

cicids_files = [
    'Monday-WorkingHours.pcap_ISCX.csv',
    'Tuesday-WorkingHours.pcap_ISCX.csv',
    'Wednesday-WorkingHours.pcap_ISCX.csv',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
    'Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
]

df_list = []
for f in cicids_files:
    fpath = os.path.join(data_path, f)
    print("loading", f)
    try:
        if 'Thursday-WorkingHours-Morning-WebAttacks' in f:
            temp = pd.read_csv(fpath, encoding="latin1")
            temp['Label'] = temp['Label'].replace({
                'Web Attack \x96 Brute Force': 'Web Attack-Brute Force',
                'Web Attack \x96 XSS': 'Web Attack-XSS',
                'Web Attack \x96 Sql Injection': 'Web Attack-Sql Injection'
            })
        else:
            temp = pd.read_csv(fpath)
        df_list.append(temp)
        print(f, "ok ->", len(temp), "rows")
    except Exception as e:
        print("problem with", f, ":", e)

if not df_list:
    raise ValueError("no files loaded, check path/names")

df = pd.concat(df_list, ignore_index=True)
print("combined shape:", df.shape)

# clean up col names
df.columns = df.columns.str.strip()
if ' Label' in df.columns:
    df.rename(columns={' Label': 'Label'}, inplace=True)

# drop obvious ID-type columns (IPs, ports, etc.)
drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
df = df.drop(columns=drop_cols, errors="ignore")
print("after dropping id cols:", df.shape)

# fix weird values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = df.shape[0]
df.dropna(inplace=True)
print("dropped", before - df.shape[0], "rows with NaNs")

# drop duplicates
before = df.shape[0]
df.drop_duplicates(inplace=True)
print("removed", before - df.shape[0], "dupes")

# split features/labels
X = df.drop('Label', axis=1)
y = df['Label']

# force numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X.dropna(inplace=True)
y = y[X.index]

print("data after cleanup:", X.shape)

# encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("classes:", le.classes_)
# --- train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

print("train size:", X_train.shape, " test size:", X_test.shape)

# --- balance classes (smote) ---
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("after smote:", X_train.shape)

# --- scale features ---
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# reshape into 3D (samples, features, 1 channel) for cnn/lstm
X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)

print("after reshape:", X_train.shape, X_test.shape)
# --- build cnn + bilstm ---
model = Sequential()
model.add(Conv1D(64, 3, activation="relu", input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(np.unique(y_enc)), activation="softmax"))

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.summary()

# --- training (just 10 epochs to test, can bump later) ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10, batch_size=64
)

# --- evaluation ---
# get predictions + basic report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("clf report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("conf matrix")
plt.show()
# shap explainability (just a small sample so it doesn’t crawl)
expl = shap.DeepExplainer(model, X_train[:200])
shap_vals = expl.shap_values(X_test[:50])
# quick summary plot
shap.summary_plot(shap_vals, X_test[:50], feature_names=X.columns)
