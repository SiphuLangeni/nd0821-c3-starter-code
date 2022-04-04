import pandas as pd
import os

from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field
from src.data import process_data
from src.model import inference


if 'DYNO' in os.environ and os.path.isdir('.dvc'):
    os.system('dvc config core.no_scm true')
    if os.system('dvc pull') != 0:
        exit('dvc pull failed')
    os.system('rm -r .dvc .apt/usr/lib/dvc')
    
app = FastAPI()

rfc_model = load('model/rfc_model.joblib.dvc')
encoder = load('model/encoder.joblib.dvc')
lb = load('model/lb.joblib.dvc')

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


@app.get('/')
async def welcome_messgage():
    return {'message': 'Welcome!!!'}

@app.post('/prediction/')
async def predict_salary(record: CensusData):

    record_dict = record.dict()

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
