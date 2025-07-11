!pip install keras-tuner tensorflow numpy pandas scikit-learn matplotlib

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras_tuner as kt

# 1️⃣ Generate Synthetic Medical Data (Age, Blood Pressure, Glucose, Cholesterol, Diagnosis)
np.random.seed(42)
num_samples = 5000
X = np.random.rand(num_samples, 4) * 100  # Normalized medical features
y = np.random.randint(0, 2, size=(num_samples,))  # 0 = Healthy, 1 = Disease

# 2️⃣ Preprocess Data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Normalize values
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3️⃣ Define Hyperparameter Tuning Function
def build_model(hp):
    model = Sequential([
        Dense(hp.Int("units_1", min_value=16, max_value=128, step=16), activation="relu", input_shape=(4,)),
        BatchNormalization(),
        Dropout(hp.Float("dropout_1", 0.2, 0.5, step=0.1)),
        Dense(hp.Int("units_2", min_value=16, max_value=64, step=16), activation="relu"),
        Dropout(hp.Float("dropout_2", 0.2, 0.5, step=0.1)),
        Dense(1, activation="sigmoid")  # Binary classification
    ])

    model.compile(
        optimizer=Adam(hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# 4️⃣ Hyperparameter Tuning with Keras Tuner
tuner = kt.RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,  # Number of different hyperparameter combinations to test
    executions_per_trial=1,
    directory="tuner_results",
    project_name="medical_fine_tuning"
)

# Search for Best Hyperparameters
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Get Best Model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# 5️⃣ Fine-Tune the Model with Best Hyperparameters
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# 6️⃣ Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Fine-Tuned Model Accuracy: {accuracy * 100:.2f}%")

# 7️⃣ Real-Time Patient Diagnosis
new_patient = np.array([[55, 130, 90, 200]])  # Example: Age, BP, Glucose, Cholesterol
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

if prediction > 0.5:
    print("⚠️ High Risk: Possible Disease Detected!")
else:
    print("✅ Low Risk: Healthy Condition")
