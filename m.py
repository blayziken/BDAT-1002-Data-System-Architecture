import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib  # For saving the model

# Load dataset
train_df = pd.read_csv("train_df.csv")

# Features & target
X = train_df.drop(columns=["readmitted"])  # Exclude target
y = train_df["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
