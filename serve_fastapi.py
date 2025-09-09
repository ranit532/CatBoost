
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
import os

# Define the input data model
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Initialize the FastAPI app
app = FastAPI(
    title="CatBoost Diabetes Prediction API",
    description="An API to predict diabetes using a CatBoost model.",
    version="1.0.0"
)

# Load the trained model
model_path = "models/catboost_model.bin"
if not os.path.exists(model_path):
    raise RuntimeError("Model not found. Please run train.py to train and save the model.")
    
model = CatBoostClassifier()
model.load_model(model_path)

@app.get("/", tags=["Root"])
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Diabetes Prediction API!"}

@app.post("/predict/", tags=["Prediction"])
def predict_diabetes(input_data: DiabetesInput):
    """
    Predicts the probability of diabetes based on input features.
    """
    # Convert input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Make a prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Return the prediction
    return {
        "prediction": "Diabetic" if prediction[0] == 1 else "Not Diabetic",
        "probability_diabetic": f"{probability[0][1]:.4f}",
        "probability_not_diabetic": f"{probability[0][0]:.4f}"
    }

# To run this API, use the command:
# uvicorn serve_fastapi:app --reload
