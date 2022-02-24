import requests

data = {
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


response = requests.post(
    url='https://predict-salary-udacity.herokuapp.com/prediction/',
    json=data
)

print(response.status_code)
print(response.json())