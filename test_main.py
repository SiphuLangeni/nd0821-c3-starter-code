
import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

@pytest.fixture
def record_sample():
    test_record_dict = {
        'age': 53,
        'workclass': 'Private',
        'fnlgt': 321865,
        'education': 'Masters', 
        'education_num': 14, 
        'marital_status': 'Married-civ-spouse', 
        'occupation': 'Exec-managerial', 
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 40,
        'native_country': 'United-States'
    }

    return test_record_dict

def test_welcome_messgage():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json() == {'message': 'Welcome!!!'}


def test_predict_salary_1(record_sample):

    record = record_sample
    r = client.post('/prediction/', json=record)

    assert r.status_code == 200
    assert r.json() == {'prediction': '>50K'}

def test_predict_salary_0(record_sample):

    record = record_sample
    record['age'] = 25
    r = client.post('/prediction/', json=record)

    assert r.status_code == 200
    assert r.json() == {'prediction': '<=50K'}
