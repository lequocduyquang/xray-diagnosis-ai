from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import pandas as pd
import mlflow

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

app = FastAPI(title="Wine Quality Prediction API")

MODEL_URI = "models:/lakehouse_local.default.wine_quality_classifier/2"
model = mlflow.pyfunc.load_model(model_uri=MODEL_URI)
print("Mô hình đã được tải thành công!")

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với API dự đoán chất lượng rượu vang!"}

@app.post("/predict")
def predict_quality(features: WineFeatures):
    feature_map = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol"
    }
    input_dict = features.dict()
    mapped_dict = {feature_map[k]: v for k, v in input_dict.items()}
    df = pd.DataFrame([mapped_dict])
    prediction = model.predict(df)
    result = "Good Quality" if prediction[0] == 1 else "Bad Quality"
    return {"predicted_quality": result}