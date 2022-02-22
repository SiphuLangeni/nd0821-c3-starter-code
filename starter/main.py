import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference





class CensusData(BaseModel):
    age: str = Field(..., example=39)
    workclass: str = Field(..., example='State-gov') 
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example='Bachelors')
    education_num: int = Field(..., example=13)
    marital_status: str = Field(..., example='Never-married')
    occupation: str = Field(..., example='Adm-clerical')
    relationship: str = Field(..., example='Not-in-family')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example='United-States')


app = FastAPI()

@app.get('/')
async def welcome_messgage():
    return {'message': 'Welcome!!!'}

@app.post('/prediction/')
async def predict_salary(record: CensusData):

    rfc_model = load('model/rfc_model.joblib')
    encoder = load('model/encoder.joblib')
    lb = load('model/lb.joblib')

    cat_features = [
        'workclass',
        'education',
        'marital_status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native_country',
    ]

    record_dict = record.dict()
    #     'age': [record.age],
    #     'workclasss': [record.workclasss],
    #     'fnlgt': [record.fnlgt],
    #     'education': [record.education], 
    #     'education_num': [record.education_num], 
    #     'marital_status': [record.marital_status], 
    #     'occupation': [record.occupation], 
    #     'relationship': [record.relationship],
    #     'race': [record.race],
    #     'sex': [record.sex],
    #     'capital_gain': [record.capital_gain],
    #     'capital_loss': [record.capital_loss],
    #     'hours_per_week': [record.hours_per_week],
    #     'native_country': [record.native_country]
    # }

    record_df = pd.DataFrame.from_dict([record_dict])

    X, _, _, _ = process_data(
        record_df,
        categorical_features=cat_features,
        encoder=encoder,
        lb=lb, 
        training=False)

    preds = inference(rfc_model, X)
    salary = '>50K' if preds[0] == 1 else '<=50K'

    return {'prediction': salary}
