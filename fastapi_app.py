from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

# Load the model using pickle
with open('D:/fastapi/ML_model_Deploy_fastapi/diabetes_model.sav', 'rb') as model_file:
    diabetes_model = pickle.load(model_file)

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post('/disease_prediction')
def predict_diabetes(input_data: ModelInput):
    input_list = [input_data.Pregnancies, input_data.Glucose, input_data.BloodPressure, input_data.SkinThickness,
                  input_data.Insulin, input_data.BMI, input_data.DiabetesPedigreeFunction, input_data.Age]
    prediction = diabetes_model.predict([input_list])
    if prediction[0] == 0:
        return {'prediction': 'The person does not have diabetes'}
    else:
        return {'prediction': 'Diabetes'}
