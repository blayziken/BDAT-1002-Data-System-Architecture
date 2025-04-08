from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("readmission_model.pkl")

# Define input schema
class PatientData(BaseModel):
    age: int
    gender: str
    primary_diagnosis: str
    num_procedures: int
    days_in_hospital: int
    comorbidity_score: int
    discharge_to: str

@app.post("/predict")
def predict_readmission(data: PatientData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"readmitted": int(prediction)}
