from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("churn_model.pkl")

@app.get("/")
def home():
    return {"message": "Churn API Running"}

@app.post("/predict")
def predict(data: dict):
    
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    
    return {
        "churn_prediction": int(pred)
    }
