import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Load training data
train_df = pd.read_csv("train_df.csv")

# Define target and features
y = train_df["readmitted"]
X = train_df.drop(columns=["readmitted"])

# Define feature types
categorical_features = ["gender", "primary_diagnosis", "discharge_to"]
numeric_features = ["age", "num_procedures", "days_in_hospital", "comorbidity_score"]

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features), # OneHotEncoder for categorical columns 
    ("num", StandardScaler(), numeric_features) # StandardScaler for numeric columns
])

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
])

# Split the dataset into training and validation sets.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, "readmission_model.pkl")
print("Model saved as readmission_model.pkl")
