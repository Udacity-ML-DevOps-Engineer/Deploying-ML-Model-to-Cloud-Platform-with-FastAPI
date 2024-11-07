from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from train.ml.data import process_data

# Initialize FastAPI app
app = FastAPI()

# Load the model and preprocessing objects
model_path = os.path.join(os.path.dirname(__file__), "model/model.pkl")
encoder_path = os.path.join(os.path.dirname(__file__), "model/encoder.pkl")
lb_path = os.path.join(os.path.dirname(__file__), "model/lb.pkl")

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int 
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the Census Salary Prediction API"}

@app.post("/predict")
async def predict(data: CensusData):
    # Convert input data to DataFrame
    df = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country
    }])

    # Define categorical features
    cat_features = ["workclass", "marital-status", "occupation", "relationship", "race", "sex"]

    # Preprocess input data using process_data function
    X, _, _, _ = process_data(df, categorical_features=cat_features, training=False, encoder=encoder, lb=lb)

    # Make prediction
    prediction = model.predict(X)
    prediction = '>50K' if prediction[0] == 1 else '<=50K'

    return {"prediction": prediction}
