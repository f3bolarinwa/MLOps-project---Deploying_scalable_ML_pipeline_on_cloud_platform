"""
Script to run sample request on API locally

Author: Femi Bolarinwa
Date: July 2023
"""

import requests

#input data to API
data = {
    'age': 52,
    'workclass': 'Self-emp-not-inc',
    'fnlgt': 209642,
    'education': 'HS-grad',
    'marital_status': 'Married-civ-spouse',
    'occupation': 'Exec-managerial',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'hoursPerWeek': 45,
    'nativeCountry': 'United-States'
}


#GET request
r = requests.get('http://127.0.0.1:8000/')
assert r.status_code == 200
print('GET Response code = {}'.format(r.status_code))
print('GET Response body = {}'.format(r.json()))


#POST request
r = requests.post('http://127.0.0.1:8000/inference', json=data)
assert r.status_code == 200
print('POST Response code = {}'.format(r.status_code))
print('POST Response body = {}'.format(r.json()))



